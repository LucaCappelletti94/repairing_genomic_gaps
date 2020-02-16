
import os
from pickle import dump, load
from dict_hash import sha256

def cache(cache_path="./cache/{function_name}/{_hash}.pkl"):
    def cache_decorator(func):
        def wrapped(*args, **kwargs):
            # Calc the hash of the parameters
            _hash = sha256({"args":args, "kwargs":kwargs})
            path = cache_path.format(_hash=_hash, function_name=func.__name__)
            # ensure that the cache folder exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            # If the file exist, load it
            if os.path.exists(path):
                print("Loading cache at {}".format(path))
                with open(path, "rb") as f:
                    return load(f)
            # else call the function and save it
            result = func(*args, **kwargs)
            with open(path, "wb") as f:
                dump(result, f)
            return result
        return wrapped
    return cache_decorator