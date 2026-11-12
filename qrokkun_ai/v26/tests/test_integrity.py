import json
import shutil
import struct
import pytest
from qrokkun_ai.v26.compare import difference, compare, Bundle
from qrokkun_ai.v26.trace import Writer, Reader, digest, json_write


def test_raw_rng_action_weight_differences(tmp_path):
    readers=[];rows=[]
    try:
        for label,value in [('a',1.),('b',2.)]:
            w=Writer(tmp_path/label,reserve=0)
            w.emit('fixture',(0,),dict(rng=bytes([int(value)]),action=int(value),weight=value))
            w.close();r=Reader(tmp_path/label);readers.append(r);rows.append(next(r.events())['data'])
        for key in ('rng','action','weight'):
            d=difference({key:rows[0][key]},{key:rows[1][key]},readers=readers)
            assert d['path']=='$.'+key
            if key!='action':assert 'byte_index' in d
    finally:
        for r in readers:r.db.close()


def test_missing_blob_and_unclosed_fail(tmp_path):
    w=Writer(tmp_path/'trace',reserve=0)
    w.emit('fixture',(0,),dict(x=b'actual'))
    with pytest.raises(ValueError):Reader(tmp_path/'trace')
    w.close()
    r=Reader(tmp_path/'trace')
    try:
        r.db.execute('PRAGMA query_only=OFF')
        # Read-only DB cannot be mutated; corrupt using the writer connection separately.
        import sqlite3
        db=sqlite3.connect(tmp_path/'trace'/'blobs.sqlite')
        db.execute('DELETE FROM blobs');db.commit();db.close()
        with pytest.raises(ValueError,match='missing blob'):list(r.events())
    finally:
        r.db.close()


@pytest.fixture(scope='module')
def complete_pair(tmp_path_factory):
    from qrokkun_ai.v26.run import main
    root=tmp_path_factory.mktemp('complete-pv26')
    for label in ('Pv26c','Pv26t'):
        main(['--arm',label,'--out-dir',str(root/label),'--trace-dir',str(root/(label+'-trace')),
              '--source-commit','a'*40,'--nonformal','--synthetic','--device','cpu','--updates','1',
              '--episodes','1','--cap','20','--validation','1','--hidden','8'])
    return root


def mutate_bundle(root,tmp_path,change):
    import gzip
    out=tmp_path/'Pv26t';shutil.copytree(root/'Pv26t',out)
    trace=tmp_path/'trace';shutil.copytree(root/'Pv26t-trace',trace)
    m=json.loads((out/'manifest.json').read_text());m['trace_dir']=str(trace);json_write(out/'manifest.json',m)
    closure=json.loads((trace/'closed.json').read_text())
    changed=False
    for shard in closure['shards']:
        path=trace/shard['name']
        with gzip.open(path,'rt') as f:rows=[json.loads(line) for line in f]
        for i,row in enumerate(rows):
            if not changed and change(row,out):changed=True
        with gzip.open(path,'wt') as f:
            for row in rows:f.write(json.dumps(row)+'\n')
        shard['sha']=digest(path.read_bytes())
    assert changed
    json_write(trace/'closed.json',closure)
    c=json.loads((out/'closed.json').read_text());c['manifest_sha']=digest((out/'manifest.json').read_bytes())
    c['trace_closure_sha']=digest((trace/'closed.json').read_bytes())
    c['checkpoints']={p.name:digest(p.read_bytes()) for p in out.glob('checkpoint-*.json')}
    json_write(out/'closed.json',c)
    return out


def test_complete_pair_and_nested_required_fields(complete_pair,tmp_path):
    assert compare(complete_pair/'Pv26c',complete_pair/'Pv26t')['full_trace_equal']
    def change(row,out):
        if row['kind']=='G':
            del row['data']['after']['P'];return True
    bad=mutate_bundle(complete_pair,tmp_path,change)
    with pytest.raises(ValueError,match='state'):compare(complete_pair/'Pv26c',bad)


@pytest.mark.parametrize('injection',['rng','action','weight'])
def test_complete_but_different(complete_pair,tmp_path,injection):
    def change(row,out):
        if injection=='rng' and row['kind']=='P' and row['key'][1]=='train':
            draw=row['data']['draws'][0];draw['pre']=draw['post'];return True
        if injection=='action' and row['kind']=='G' and row['key'][1]=='train':
            row['data']['action']=(row['data']['action']+1)%9;return True
        if injection=='weight' and row['kind']=='checkpoint' and row['key']==[1]:
            model=row['data']['model'];name='policy_head.bias'
            import sqlite3,zlib
            ref=dict(model[name]); db=sqlite3.connect(out.parent/'trace'/'blobs.sqlite')
            raw=bytearray(zlib.decompress(db.execute('SELECT raw FROM blobs WHERE sha=?',(ref['blob'],)).fetchone()[0]))
            raw[0]^=1; ref['blob']=digest(raw)
            db.execute('INSERT INTO blobs VALUES (?,?,?)',(ref['blob'],len(raw),zlib.compress(raw)));db.commit();db.close()
            model[name]=ref
            json_write(out/'checkpoint-01.json',row['data']);return True
    changed=mutate_bundle(complete_pair,tmp_path,change)
    if injection in ('rng','action'):
        with pytest.raises(ValueError):compare(complete_pair/'Pv26c',changed)
        return
    result=compare(complete_pair/'Pv26c',changed)
    assert result['status']=='complete_different'
    assert result['first_difference'] is not None
    assert result['weights_bitwise_equal'] == (injection!='weight')


def test_missing_named_weight_failclosed(complete_pair,tmp_path):
    def change(row,out):
        if row['kind']=='checkpoint' and row['key']==[1]:
            del row['data']['model']['policy_head.bias']
            json_write(out/'checkpoint-01.json',row['data']);return True
    bad=mutate_bundle(complete_pair,tmp_path,change)
    with pytest.raises(ValueError,match='model'):compare(complete_pair/'Pv26c',bad)
