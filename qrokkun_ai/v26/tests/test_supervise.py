import json
from pathlib import Path
import sys
import pytest
from qrokkun_ai.v26.supervise import supervise


def test_sequential_and_difference_no_retry(tmp_path):
    path=tmp_path/'order'
    def cmd(label,code=0):
        return [sys.executable,'-c',f'from pathlib import Path; p=Path({str(path)!r}); p.open("a").write({label!r}); raise SystemExit({code})']
    result=supervise(tmp_path,{'Pv26c':cmd('c'),'Pv26t':cmd('t'),'compare':cmd('x',2)})
    assert path.read_text()=='ctx'
    assert result['exitcodes']=={'Pv26c':0,'Pv26t':0,'compare':2}
    assert result['status']=='complete_different'


@pytest.mark.parametrize("stop_after",["Pv26c","Pv26t"])
def test_stop_between_arms(tmp_path,stop_after):
    path=tmp_path/'must-not-exist'
    cmds={'Pv26c':[sys.executable,'-c','pass'],'Pv26t':[sys.executable,'-c','pass'],'compare':[sys.executable,'-c',f'open({str(path)!r},"w")']}
    result=supervise(tmp_path,cmds,after_child=lambda label,stop:stop() if label==stop_after else None)
    assert not path.exists()
    assert result['status']=='stopped'
    assert result['exitcodes']==({'Pv26c':0} if stop_after=='Pv26c' else {'Pv26c':0,'Pv26t':0})


def test_signal_reaps_child_and_skips_next(tmp_path):
    cmds={'Pv26c':[sys.executable,'-c','import os,signal,time; os.kill(os.getppid(),signal.SIGTERM); time.sleep(20)'],
          'Pv26t':[sys.executable,'-c','raise SystemExit(99)'],'compare':[sys.executable,'-c','raise SystemExit(99)']}
    result=supervise(tmp_path,cmds)
    assert result['status']=='stopped'
    assert set(result['exitcodes'])=={'Pv26c'}
    assert result['exitcodes']['Pv26c'] != 0
