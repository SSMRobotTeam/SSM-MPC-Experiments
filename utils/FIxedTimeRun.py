import time

from utils.timing_utils import fixed_cpu_time


@fixed_cpu_time(target_cpu_seconds=0.07)
def my_task(loop_idx: int):
    """Small timing decorator demo."""
    print(f"task {loop_idx}: running...")
    return sum(j * 3 for j in range(2))


if __name__ == "__main__":
    actual_cpu_old = time.process_time()
    print(f"start CPU time = {actual_cpu_old:.6f}s\n")

    for i in range(10):
        my_task(i + 1)
        actual_cpu = time.process_time()
        print(f"task {i + 1}: CPU interval = {actual_cpu - actual_cpu_old:.6f}s\n")
        actual_cpu_old = actual_cpu
