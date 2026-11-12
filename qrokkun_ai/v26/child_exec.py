"""Linux launch trampoline: restore the caller mask before exec or heavy imports."""
import os
import signal
import sys

if __name__=='__main__':
    mask=[int(x) for x in sys.argv[1].split(',') if x]
    signal.pthread_sigmask(signal.SIG_SETMASK,mask)
    os.execvp(sys.argv[2],sys.argv[2:])
