from __future__ import annotations

import time
from typing import List, Sequence

from src.distributed import distributed_multiply
from src.matrix_ops import generate_matrix, matrices_equal, multiply_serial
from src.models import BenchmarkResult, MatrixTest


def run_benchmarks(
    tests: Sequence[MatrixTest],
    workers: Sequence[tuple[str, int]],
    low: int = -9,
    high: int = 9,
) -> List[BenchmarkResult]:
    results: List[BenchmarkResult] = []

    for index, test in enumerate(tests, start=1):
        print(
            f"[benchmark] teste {index}: A={test.a.rows}x{test.a.cols} "
            f"B={test.b.rows}x{test.b.cols}",
            flush=True,
        )
        a = generate_matrix(test.a.rows, test.a.cols, low=low, high=high)
        b = generate_matrix(test.b.rows, test.b.cols, low=low, high=high)

        t0 = time.perf_counter()
        serial_result = multiply_serial(a, b)
        t1 = time.perf_counter()

        t2 = time.perf_counter()
        distributed_result = distributed_multiply(a, b, workers)
        t3 = time.perf_counter()

        serial_ms = (t1 - t0) * 1000.0
        distributed_ms = (t3 - t2) * 1000.0
        speedup = serial_ms / distributed_ms if distributed_ms > 0 else float("inf")
        correct = matrices_equal(serial_result, distributed_result)

        print(
            f"[benchmark] teste {index} finalizado: "
            f"serial={serial_ms:.4f}ms distribuída={distributed_ms:.4f}ms "
            f"correto={correct}",
            flush=True,
        )

        results.append(
            BenchmarkResult(
                test_index=index,
                a_shape=f"{test.a.rows}x{test.a.cols}",
                b_shape=f"{test.b.rows}x{test.b.cols}",
                workers=len(workers),
                serial_ms=serial_ms,
                distributed_ms=distributed_ms,
                speedup=speedup,
                correct=correct,
            )
        )

    return results
