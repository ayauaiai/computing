"""
Experiment A — The Granularity Trap
Compares fine-grained tasks (chunksize=1) with coarse-grained tasks
(chunksize=10_000) for P = 1, 2, 4, 8 workers.
"""
import argparse
import csv
import time
from multiprocessing import Pool


def collatz_steps(n: int) -> int:
    steps = 0
    while n > 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        steps += 1
    return steps


def run_granularity(workers: int, total_items: int, chunk_size: int):
    numbers = range(1, total_items + 1)
    start = time.perf_counter()
    with Pool(processes=workers) as pool:
        results = pool.map(collatz_steps, numbers, chunksize=chunk_size)
    elapsed = time.perf_counter() - start
    return elapsed, sum(results)


def karp_flatt(t1: float, tp: float, p: int):
    if p == 1:
        return None
    return ((p * tp / t1) - 1.0) / (p - 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=int, default=1_000_000)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--chunks", type=int, nargs="+", default=[1, 10_000])
    parser.add_argument("--csv", default="exp_a_results.csv")
    args = parser.parse_args()

    rows = []
    for chunk in args.chunks:
        print(f"\n=== Experiment A: chunksize={chunk:,} ===")
        baseline = None
        measurements = {}
        for p in args.workers:
            elapsed, checksum = run_granularity(p, args.items, chunk)
            measurements[p] = elapsed
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
                "chunksize": chunk, "workers": p, "time_seconds": elapsed,
                "speedup": speedup, "karp_flatt": e, "checksum": checksum
            })

    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved measurements to {args.csv}")


if __name__ == "__main__":
    main()
