"""v32 lifecycle only; numerical pool methods remain inherited from v31."""
import sys
from qrokkun_ai.v31 import core


class EpisodePool(core.EpisodePool):
    def close(self):
        if self._closed:
            return
        if not hasattr(self, '_cleanup_errors'):
            self._cleanup_errors = []
            self._cleanup_done = set()
        errors, done = self._cleanup_errors, self._cleanup_done

        def attempt(key, operation):
            if key in done:
                return True
            try:
                operation()
            except BaseException as exc:
                errors.append(RuntimeError(f'{key}: {exc!r}'))
                return False
            done.add(key)
            return True

        attempt('stop', self.stop_event.set)
        for i, (conn, proc) in enumerate(self.children):
            attempt(('connection', i), conn.close)
        for i, (_, proc) in enumerate(self.children):
            if ('process', i) in done:
                continue
            # Each escalation and reap is independent: a failed TERM must not
            # prevent KILL or its bounded join. No retries inside this call.
            def alive():
                try:
                    return proc.is_alive()
                except BaseException as exc:
                    errors.append(RuntimeError(f'worker {i} is_alive: {exc!r}'))
                    return True
            attempt(('join', i), lambda: proc.join(timeout=1.0))
            if alive():
                attempt(('terminate', i), proc.terminate)
                attempt(('term_join', i), lambda: proc.join(timeout=2.0))
            if alive():
                attempt(('kill', i), proc.kill)
                attempt(('kill_join', i), lambda: proc.join(timeout=2.0))
            if alive():
                errors.append(RuntimeError(f'worker {i} still live after cleanup'))
            else:
                attempt(('process', i), proc.close)
        for name in ('w_shm', 't_shm'):
            shm = getattr(self, name, None)
            if shm is None:
                continue
            attempt((name, 'close'), shm.close)
            def unlink():
                try:
                    shm.unlink()
                except FileNotFoundError:
                    pass
            attempt((name, 'unlink'), unlink)
        if errors:
            primary = sys.exception()
            evidence = list(errors)
            if primary is not None:
                evidence.insert(0, primary)
            raise BaseExceptionGroup('v32 pool cleanup failed: ' + repr(errors), evidence)
        self._closed = True
