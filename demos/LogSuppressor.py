import subprocess as sp

class LogSuppressor:
    def __init__(self):
        pass
        

    @staticmethod
    def suppress_subprocess_logging():
        original_call = sp.call

        def _call_nostderr(*args, **kwargs):
            kwargs['stderr'] = sp.DEVNULL
            kwargs['stdout'] = sp.DEVNULL
            return original_call(*args, **kwargs)

        sp.call = _call_nostderr
