"""CLI-only exception translation; Python main APIs retain their exceptions."""
import signal
import traceback
from contextlib import contextmanager


@contextmanager
def owner_signals():
    handlers={}
    def stop(sig,frame):raise InterruptedError('owner stop')
    try:
        for sig in (signal.SIGINT,signal.SIGTERM):handlers[sig]=signal.signal(sig,stop)
        yield
    finally:
        for sig,handler in handlers.items():signal.signal(sig,handler)


def boundary(main,argv=None):
    try:
        with owner_signals():return main(argv)
    except (InterruptedError,KeyboardInterrupt):
        traceback.print_exc()
        return 130
    except SystemExit as e:
        # Only argparse usage failures share the scientific difference code.
        # Preserve help and explicit SystemExit control flow from other callers.
        tb=e.__traceback__
        while e.code==2 and tb is not None:
            if (tb.tb_frame.f_globals.get('__name__')=='argparse'
                    and tb.tb_frame.f_code.co_name=='error'):
                return 3
            tb=tb.tb_next
        raise
    except Exception:
        traceback.print_exc()
        return 3
