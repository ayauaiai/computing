"""
Experiment B — The Inter-Process Queue Choke-Point
Each worker sends one queue message per item. This intentionally creates
high IPC traffic. Run only on a machine where you can tolerate a slow test.
"""
import argparse
import csv
import time
from multiprocessing import Process, Queue


def queue_worker(out_queue, items_per_worker: int):
    for i in range(items_per_worker):
        out_queue.put(i % 7)


def run_queue_benchmark(workers: int, total_items: int):
    q = Queue()
    items_per_worker = total_items // workers
    remainder = total_items % workers

    processes = []
    for worker_id in range(workers):
        count = items_per_worker + (1 if worker_id < remainder else 0)
        processes.append(Process(target=queue_worker, args=(q, count)))

    start = time.perf_counter()
    for process in processes:
        process.start()

    collected = 0
    checksum = 0
    while collected < total_items:
        value = q.get()
        checksum += value
        collected += 1

    for process in processes:
        process.join()

    elapsed = time.perf_counter() - start
    q.close()
    q.join_thread()
    return elapsed, checksum


def karp_flatt(t1: float, tp: float, p: int):
    if p == 1:
        return None
    return ((p * tp / t1) - 1.0) / (p - 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=int, default=400_000)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--csv", default="exp_b_results.csv")
    args = parser.parse_args()

    rows = []
    baseline = None
    print("\n=== Experiment B: Queue Contention ===")
    for p in args.workers:
        elapsed, checksum = run_queue_benchmark(p, args.items)
        if p == 1:
            baseline = elapsed
        speedup = baseline / elapsed if baseline is not None else None
        e = karp_flatt(baseline, elapsed, p) if baseline is not None else None
        print(
            f"P={p:2d} | Time={elapsed:8.3f}s | "
            f"Speedup={speedup:6.3f}x | "
            f"Karp-Flatt={e if e is not None else 'N/A'} | "
            f"Checksum={checksum}"
        )
        rows.append({
            "workers": p, "time_seconds": elapsed, "speedup": speedup,
            "karp_flatt": e, "checksum": checksum
        })

    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved measurements to {args.csv}")


if __name__ == "__main__":
    main()
