import signal
import time


class TimeLimit:
    def __init__(self, seconds: int):
        self.seconds = seconds
        self._start = None

    def __enter__(self):
        self._start = time.time()
        signal.signal(signal.SIGALRM, self._raise_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, exc_type, exc_value, traceback):
        signal.alarm(0)

    def _raise_timeout(self, signum, frame):
        raise TimeoutError(f"Timeout after {self.seconds} seconds")
