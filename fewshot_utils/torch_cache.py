import hashlib
from functools import wraps
from pathlib import Path
from typing import Callable, Union

import torch


def torch_cache(path_provider: Union[Path, str, Callable]):
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # check if path_provider is a function
            if callable(path_provider):
                cache_file = Path(path_provider(*args))
            else:
                cache_file = Path(path_provider)

            func_name = func.__name__
            str_args = [arg for arg in args if isinstance(arg, str)]

            cache_key = "".join([func_name] + str_args).encode()

            # hash the arguments and append the hash to the filename
            hasher = hashlib.sha256()
            hasher.update(cache_key)
            arg_hash = hasher.hexdigest()
            cache_file = cache_file.with_name(cache_file.stem + "_" + arg_hash + cache_file.suffix)

            if cache_file.exists():
                print(f"Loading cached result from {cache_file}")
                result = torch.load(cache_file)
            else:
                print(f"No cached result, running {func.__name__}...")
                result = func(*args, **kwargs)
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                torch.save(result, cache_file)
                print(f"Saved result to {cache_file}")
            return result

        return wrapper

    return decorator
