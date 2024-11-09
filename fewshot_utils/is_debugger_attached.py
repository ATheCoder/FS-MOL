import sys


def is_debugger_attached():
    return sys.gettrace() is not None
