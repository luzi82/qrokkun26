"""Streaming gzip event shards and on-disk compressed content-addressed raw bytes.

No pickle deserialization is needed by the comparator. SQLite is the bounded-memory
blob index; each blob is individually losslessly compressed. Events own references,
never live tensor views. The main thread alone writes canonical worker buffers.
"""
import gzip
import ast
import math
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import struct
import zlib
import numpy as np
import torch


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def json_write(path, data):
    path = Path(path)
    with path.with_suffix('.new').open('w') as f:
        json.dump(data, f, sort_keys=True, allow_nan=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    path.with_suffix('.new').replace(path)


class Writer:
    def __init__(self, root, *, reserve=1024**3, shard_bytes=8*1024**2):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.reserve = reserve
        self.shard_bytes = shard_bytes
        self.db = sqlite3.connect(self.root/'blobs.sqlite')
        self.db.execute('CREATE TABLE blobs (sha TEXT PRIMARY KEY, size INTEGER, raw BLOB)')
        self.files = []
        self.stream = None
        self.count = self.size = 0
        self.counts = {}
        self.closed = False

    def watermark(self):
        if shutil.disk_usage(self.root).free < self.reserve:
            raise OSError('trace disk watermark reached; preserving incomplete trace')

    def blob(self, raw, dtype, shape):
        sha = digest(raw)
        if self.db.execute('SELECT 1 FROM blobs WHERE sha=?',(sha,)).fetchone() is None:
            self.db.execute('INSERT INTO blobs VALUES (?, ?, ?)',
                            (sha, len(raw), zlib.compress(raw, 1)))
        return {'blob': sha, 'dtype': dtype, 'shape': list(shape), 'bytes': len(raw)}

    def encode(self, x):
        if isinstance(x, torch.Tensor):
            t = x.detach().cpu().contiguous()
            return self.blob(t.reshape(-1).view(torch.uint8).numpy().tobytes(), str(t.dtype), t.shape)
        if isinstance(x, np.ndarray):
            return self.blob(x.tobytes(order='C'), 'numpy:'+ (repr(x.dtype.descr) if x.dtype.fields else x.dtype.str), x.shape)
        if isinstance(x, (float, np.floating)):
            return self.blob(struct.pack('>d', float(x)), 'python.float64.be', [])
        if isinstance(x, bytes):
            return self.blob(x, 'bytes', [len(x)])
        if isinstance(x, dict):
            return {str(k): self.encode(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [self.encode(v) for v in x]
        if isinstance(x, np.integer):
            return int(x)
        if x is None or isinstance(x, (str, int, bool)):
            return x
        raise TypeError(type(x))

    def emit(self, kind, key, data, audit=None):
        self.watermark()
        row = {'kind': kind, 'key': list(key), 'data': self.encode(data), 'audit': audit or {}}
        raw = (json.dumps(row, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()
        if self.stream is None or self.size >= self.shard_bytes:
            self._finish_shard()
            name = f'events-{len(self.files):06d}.gz'
            self.stream = gzip.open(self.root/name, 'xb', compresslevel=1)
            self.name, self.size = name, 0
        self.stream.write(raw)
        self.size += len(raw)
        self.count += 1
        self.counts[kind] = self.counts.get(kind, 0)+1
        # Persist every event: abrupt stop leaves readable prefixes and committed blobs.
        self.stream.flush()
        self.db.commit()

    def _finish_shard(self):
        if self.stream:
            self.stream.close()
            p = self.root/self.name
            with p.open('rb') as f:
                os.fsync(f.fileno())
            self.files.append({'name': self.name, 'sha': digest(p.read_bytes())})
            self.stream = None

    def close(self):
        if self.closed:
            return
        self._finish_shard()
        self.db.commit()
        self.db.close()
        json_write(self.root/'closed.json', {'events': self.count, 'counts': self.counts, 'shards': self.files})
        self.closed = True

    def interrupt(self):
        self._finish_shard()
        self.db.commit()
        self.db.close()
        self.closed = True  # deliberately no closed.json


class Reader:
    def __init__(self, root):
        self.root = Path(root)
        try:
            self.closure = json.loads((self.root/'closed.json').read_text())
        except (OSError, ValueError) as e:
            raise ValueError('missing/invalid trace closure') from e
        self.db = sqlite3.connect(f'file:{self.root / "blobs.sqlite"}?mode=ro', uri=True)

    def raw(self, ref):
        shape=ref['shape']
        if not isinstance(shape,list) or any(type(n)!=int or n<0 for n in shape):
            raise ValueError('invalid blob shape')
        dtype=ref['dtype']
        if dtype=='python.float64.be':size=8
        elif dtype=='bytes':size=1
        elif dtype.startswith('torch.'):
            sizes={'bool':1,'uint8':1,'int8':1,'int16':2,'float16':2,'bfloat16':2,'int32':4,'float32':4,'int64':8,'float64':8,'complex64':8,'complex128':16}
            if dtype[6:] not in sizes:raise ValueError('unsupported tensor dtype')
            size=sizes[dtype[6:]]
        elif dtype.startswith('numpy:'):
            spec=dtype[6:]
            size=np.dtype(ast.literal_eval(spec) if spec.startswith('[') else spec).itemsize
        else:raise ValueError('invalid blob dtype')
        if math.prod(shape)*size!=ref['bytes']:raise ValueError('blob shape/byte length mismatch')
        row = self.db.execute('SELECT size, raw FROM blobs WHERE sha=?', (ref['blob'],)).fetchone()
        if row is None:
            raise ValueError('missing blob '+ref['blob'])
        raw = zlib.decompress(row[1])
        if len(raw) != row[0] or len(raw) != ref['bytes'] or digest(raw) != ref['blob']:
            raise ValueError('corrupt blob '+ref['blob'])
        return raw

    def validate(self, x):
        if isinstance(x, dict):
            if {'blob','dtype','shape','bytes'} & set(x):
                if set(x) != {'blob','dtype','shape','bytes'}:
                    raise ValueError('invalid raw reference')
                self.raw(x)
            else:
                for v in x.values():
                    self.validate(v)
        elif isinstance(x, list):
            for v in x:
                self.validate(v)

    def events(self):
        count, counts = 0, {}
        for spec in self.closure['shards']:
            p = self.root/spec['name']
            h = hashlib.sha256()
            with p.open('rb') as f:
                for b in iter(lambda: f.read(1024**2), b''):
                    h.update(b)
            if h.hexdigest() != spec['sha']:
                raise ValueError('corrupt event shard '+spec['name'])
            with gzip.open(p, 'rt') as f:
                for line in f:
                    row = json.loads(line)
                    if set(row) != {'kind','key','data','audit'}:
                        raise ValueError('invalid event envelope')
                    self.validate(row['data'])
                    count += 1
                    counts[row['kind']] = counts.get(row['kind'], 0)+1
                    yield row
        if count != self.closure['events'] or counts != self.closure['counts']:
            raise ValueError('incomplete trace event counts')
