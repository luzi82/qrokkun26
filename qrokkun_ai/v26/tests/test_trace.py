import struct
import torch
import pytest
from qrokkun_ai.v26.trace import Writer, Reader


def test_bits_cas_and_truncation(tmp_path):
    root = tmp_path/'trace'
    w = Writer(root, reserve=0)
    raw = bytes.fromhex('00000000000000800100c07f')
    x = torch.frombuffer(bytearray(raw), dtype=torch.float32)
    w.emit('fixture', (0,), dict(x=x, f=-0.0))
    w.emit('fixture', (1,), dict(x=x))
    w.close()
    r = Reader(root)
    rows = list(r.events())
    assert r.raw(rows[0]['data']['x']) == raw
    assert r.raw(rows[0]['data']['f']) == struct.pack('>d', -0.0)
    assert rows[0]['data']['x'] == rows[1]['data']['x']
    with (root/'events-000000.gz').open('ab') as f:
        f.write(b'broken')
    with pytest.raises((ValueError, OSError, EOFError)):
        list(Reader(root).events())


def test_descriptor_shape_failclosed(tmp_path):
    w=Writer(tmp_path/'trace',reserve=0)
    w.emit('fixture',(0,),dict(x=torch.ones(2)))
    w.close();r=Reader(tmp_path/'trace')
    ref=dict(next(r.events())['data']['x']);ref['shape']=[3]
    with pytest.raises(ValueError,match='shape'):r.raw(ref)
