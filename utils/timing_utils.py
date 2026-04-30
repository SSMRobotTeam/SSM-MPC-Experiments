import functools
import time
from typing import Callable


def fixed_cpu_time(target_cpu_seconds: float) -> Callable:
    """Decorator that keeps a function's CPU time close to a fixed period."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_cpu = time.process_time()
            result = func(*args, **kwargs)
            elapsed_cpu = time.process_time() - start_cpu

            if elapsed_cpu < target_cpu_seconds:
                end_cpu = start_cpu + target_cpu_seconds
                while time.process_time() < end_cpu:
                    pass
            else:
                print(
                    f"Warning: {func.__name__} CPU time "
                    f"({elapsed_cpu:.6f}s) exceeded target "
                    f"({target_cpu_seconds:.6f}s)"
                )

            return result

        return wrapper

    return decorator
