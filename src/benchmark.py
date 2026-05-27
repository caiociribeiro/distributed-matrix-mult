import time

from .distributed import (
    distributed_multiply_parallel,
    distributed_multiply_serial,
)
from .matrix_ops import (
    estimate_work_units,
    generate_matrix,
    matrices_equal,
    multiply_parallel_processes,
    multiply_parallel_threads,
    multiply_serial,
)
from .models import BenchmarkResult
from .report import save_test_artifacts


# executa uma funcao e retorna resultado + tempo em ms
def measure(func, *args):
    t0 = time.perf_counter()
    result = func(*args)
    t1 = time.perf_counter()

    return result, (t1 - t0) * 1000


def add_result(
    results,
    test_index,
    mode,
    workers,
    test,
    work_units,
    elapsed_ms,
    serial_ms,
    correct,
):
    results.append(
        BenchmarkResult(
            test_index=test_index,
            mode=mode,
            workers=workers,
            rows_a=test.a.rows,
            cols_a=test.a.cols,
            rows_b=test.b.rows,
            cols_b=test.b.cols,
            work_units=work_units,
            elapsed_ms=elapsed_ms,
            speedup=serial_ms / elapsed_ms,
            correct=correct,
        )
    )


def run_benchmarks(
    tests,
    distributed_workers,
    distributed_worker_counts,
    local_parallel_workers=4,
    low=-9,
    high=9,
    out_dir=None,
):
    results = []

    for index, test in enumerate(tests, start=1):
        print(
            f"[benchmark] teste {index}: "
            f"A={test.a.rows}x{test.a.cols} "
            f"B={test.b.rows}x{test.b.cols}"
        )

        a = generate_matrix(test.a.rows, test.a.cols, low, high)
        b = generate_matrix(test.b.rows, test.b.cols, low, high)

        work_units = estimate_work_units(a, b)

        outputs = {}

        print("[benchmark] executando multiplicacao serial")
        serial_result, serial_ms = measure(multiply_serial, a, b)
        print("[benchmark] serial finalizado")
        outputs["serial"] = serial_result

        add_result(
            results,
            index,
            "serial",
            1,
            test,
            work_units,
            serial_ms,
            serial_ms,
            True,
        )

        print("[benchmark] executando multiplicacao com threads")
        threads_result, threads_ms = measure(
            multiply_parallel_threads,
            a,
            b,
            local_parallel_workers,
        )
        print("[benchmark] threads finalizado")
        outputs["threads"] = threads_result

        add_result(
            results,
            index,
            "threads",
            local_parallel_workers,
            test,
            work_units,
            threads_ms,
            serial_ms,
            matrices_equal(serial_result, threads_result),
        )

        print("[benchmark] executando multiplicacao com processos")
        processes_result, processes_ms = measure(
            multiply_parallel_processes,
            a,
            b,
            local_parallel_workers,
        )
        print("[benchmark] processos finalizada")
        outputs["processes"] = processes_result

        add_result(
            results,
            index,
            "processes",
            local_parallel_workers,
            test,
            work_units,
            processes_ms,
            serial_ms,
            matrices_equal(serial_result, processes_result),
        )

        print("[benchmark] executando multiplicacao distribuida")
        for worker_count in distributed_worker_counts:
            workers = distributed_workers[:worker_count]

            print(f"[benchmark] distribuido serial com {worker_count} worker(s)")
            dist_serial_result, dist_serial_ms = measure(
                distributed_multiply_serial,
                a,
                b,
                workers,
            )
            outputs[f"distributed_serial_{worker_count}"] = dist_serial_result

            add_result(
                results,
                index,
                "distributed_serial",
                worker_count,
                test,
                work_units,
                dist_serial_ms,
                serial_ms,
                matrices_equal(serial_result, dist_serial_result),
            )
            print("[benchmark] distribuido serial finalizado")

            print(f"[benchmark] distribuido paralelo com {worker_count} worker(s) ")

            dist_parallel_result, dist_parallel_ms = measure(
                distributed_multiply_parallel,
                a,
                b,
                workers,
            )

            outputs[f"distributed_parallel_{worker_count}"] = dist_parallel_result

            add_result(
                results,
                index,
                "distributed_parallel",
                worker_count,
                test,
                work_units,
                dist_parallel_ms,
                serial_ms,
                matrices_equal(serial_result, dist_parallel_result),
            )

            print(
                f"[benchmark] distribuido paralelo com {worker_count} worker(s) finalizado"
            )

        if out_dir:
            save_test_artifacts(
                out_dir,
                index,
                a,
                b,
                outputs,
            )

        print(f"[benchmark] teste {index} finalizado")

    return results
