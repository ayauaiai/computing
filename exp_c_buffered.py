"""
Experiment C — Local Buffer Chunking
Workers accumulate values locally and send one aggregated sum per batch,
rather than sending every individual item through the shared queue.
"""
import argparse
import csv
import time
from multiprocessing import Process, Queue


def buffered_worker(out_queue, items_per_worker: int, batch_size: int):
    buffer = []
    for i in range(items_per_worker):
        buffer.append(i % 7)
        if len(buffer) >= batch_size:
            out_queue.put(sum(buffer))
            buffer.clear()

    if buffer:
        out_queue.put(sum(buffer))
    out_queue.put(None)  # Sentinel: this worker has finished


def run_buffered(workers: int, total_items: int, batch_size: int):
    q = Queue()
    items_per_worker = total_items // workers
    remainder = total_items % workers

    processes = []
    for worker_id in range(workers):
        count = items_per_worker + (1 if worker_id < remainder else 0)
        processes.append(
            Process(target=buffered_worker, args=(q, count, batch_size))
        )

    start = time.perf_counter()
    for process in processes:
        process.start()

    total_sum = 0
    active_workers = workers
    while active_workers > 0:
        value = q.get()
        if value is None:
            active_workers -= 1
        else:
            total_sum += value

    for process in processes:
        process.join()

    elapsed = time.perf_counter() - start
    q.close()
    q.join_thread()
    return elapsed, total_sum


def karp_flatt(t1: float, tp: float, p: int):
    if p == 1:
        return None
    return ((p * tp / t1) - 1.0) / (p - 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=int, default=10_000_000)
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--csv", default="exp_c_results.csv")
    args = parser.parse_args()

    rows = []
    baseline = None
    print("\n=== Experiment C: Buffered Queue ===")
    for p in args.workers:
        elapsed, checksum = run_buffered(p, args.items, args.batch_size)
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
