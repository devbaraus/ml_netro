from functools import wraps
from time import time

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('### func:{} \n### args:[{}, {}] \n### took: {:2.4f} sec'.format(f.__name__, args, kw, te-ts))
        return result
    return wrap