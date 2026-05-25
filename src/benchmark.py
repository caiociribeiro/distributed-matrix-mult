from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

from .distributed import distributed_multiply_parallel, distributed_multiply_serial
from .matrix_ops import (
    estimate_work_units,
    generate_matrix,
    matrices_equal,
    multiply_parallel_processes,
    multiply_parallel_threads,
    multiply_serial,
)
from .models import BenchmarkResult, MatrixTest
from .report import save_test_artifacts


def run_benchmarks(
    tests: Sequence[MatrixTest],
    distributed_workers: Sequence[tuple[str, int]],
    local_parallel_workers: int = 4,
    low: int = -9,
    high: int = 9,
    out_dir: Path | None = None,
) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []

    for index, test in enumerate(tests, start=1):
        print(
            f"[benchmark] teste {index}: A={test.a.rows}x{test.a.cols} "
            f"B={test.b.rows}x{test.b.cols}",
            flush=True,
        )

        a = generate_matrix(test.a.rows, test.a.cols, low=low, high=high)
        b = generate_matrix(test.b.rows, test.b.cols, low=low, high=high)
        work_units = estimate_work_units(a, b)

        outputs: dict[str, object] = {}

        t0 = time.perf_counter()
        serial_result = multiply_serial(a, b)
        t1 = time.perf_counter()
        serial_ms = (t1 - t0) * 1000.0
        outputs["serial"] = serial_result

        t0 = time.perf_counter()
        threads_result = multiply_parallel_threads(a, b, workers=local_parallel_workers)
        t1 = time.perf_counter()
        threads_ms = (t1 - t0) * 1000.0
        outputs["threads"] = threads_result

        t0 = time.perf_counter()
        processes_result = multiply_parallel_processes(a, b, workers=local_parallel_workers)
        t1 = time.perf_counter()
        processes_ms = (t1 - t0) * 1000.0
        outputs["processes"] = processes_result

        results.append(
            BenchmarkResult(
                test_index=index,
                mode="serial",
                workers=1,
                rows_a=test.a.rows,
                cols_a=test.a.cols,
                rows_b=test.b.rows,
                cols_b=test.b.cols,
                work_units=work_units,
                elapsed_ms=serial_ms,
                speedup=1.0,
                correct=True,
            )
        )

        results.append(
            BenchmarkResult(
                test_index=index,
                mode="threads",
                workers=local_parallel_workers,
                rows_a=test.a.rows,
                cols_a=test.a.cols,
                rows_b=test.b.rows,
                cols_b=test.b.cols,
                work_units=work_units,
                elapsed_ms=threads_ms,
                speedup=serial_ms / threads_ms if threads_ms > 0 else float("inf"),
                correct=matrices_equal(serial_result, threads_result),
            )
        )

        results.append(
            BenchmarkResult(
                test_index=index,
                mode="processes",
                workers=local_parallel_workers,
                rows_a=test.a.rows,
                cols_a=test.a.cols,
                rows_b=test.b.rows,
                cols_b=test.b.cols,
                work_units=work_units,
                elapsed_ms=processes_ms,
                speedup=serial_ms / processes_ms if processes_ms > 0 else float("inf"),
                correct=matrices_equal(serial_result, processes_result),
            )
        )

        for worker_count in sorted({len(distributed_workers)} | {2, 4, 6}):
            workers = list(distributed_workers[:worker_count])
            if len(workers) < worker_count:
                continue

            t0 = time.perf_counter()
            dist_serial_result = distributed_multiply_serial(a, b, workers)
            t1 = time.perf_counter()
            dist_serial_ms = (t1 - t0) * 1000.0
            outputs[f"distributed_serial_{worker_count}"] = dist_serial_result

            t0 = time.perf_counter()
            dist_parallel_result = distributed_multiply_parallel(
                a,
                b,
                workers,
            )
            t1 = time.perf_counter()
            dist_parallel_ms = (t1 - t0) * 1000.0
            outputs[f"distributed_parallel_{worker_count}"] = dist_parallel_result

            results.append(
                BenchmarkResult(
                    test_index=index,
                    mode="distributed_serial",
                    workers=worker_count,
                    rows_a=test.a.rows,
                    cols_a=test.a.cols,
                    rows_b=test.b.rows,
                    cols_b=test.b.cols,
                    work_units=work_units,
                    elapsed_ms=dist_serial_ms,
                    speedup=serial_ms / dist_serial_ms if dist_serial_ms > 0 else float("inf"),
                    correct=matrices_equal(serial_result, dist_serial_result),
                )
            )

            results.append(
                BenchmarkResult(
                    test_index=index,
                    mode="distributed_parallel",
                    workers=worker_count,
                    rows_a=test.a.rows,
                    cols_a=test.a.cols,
                    rows_b=test.b.rows,
                    cols_b=test.b.cols,
                    work_units=work_units,
                    elapsed_ms=dist_parallel_ms,
                    speedup=serial_ms / dist_parallel_ms if dist_parallel_ms > 0 else float("inf"),
                    correct=matrices_equal(serial_result, dist_parallel_result),
                )
            )

        if out_dir is not None:
            save_test_artifacts(
                out_dir=out_dir,
                test_index=index,
                a=a,
                b=b,
                outputs_by_mode={k: v for k, v in outputs.items() if hasattr(v, "shape")},
                metadata={
                    "local_parallel_workers": local_parallel_workers,
                    "distributed_workers": [list(x) for x in distributed_workers],
                },
            )

        print(f"[benchmark] teste {index} finalizado", flush=True)

    return results
