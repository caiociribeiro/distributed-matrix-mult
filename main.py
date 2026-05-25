from __future__ import annotations

import multiprocessing as mp
import sys
import time

from src.benchmark import run_benchmarks
from src.distributed import shutdown_workers
from src.parser import parse_test_file
from src.report import (
    ensure_output_dir,
    plot_results,
    print_table,
    write_csv,
)
from src.worker import run_worker_server

BASE_PORT = 5000
DISTRIBUTED_WORKER_COUNTS = [2, 4, 6]
LOCAL_WORKERS = 4


def main() -> int:
    if len(sys.argv) < 2:
        print("Uso: python main.py <arquivo_de_testes>", file=sys.stderr)
        return 2

    tests = parse_test_file(sys.argv[1])
    workers = [
        ("127.0.0.1", BASE_PORT + i) for i in range(max(DISTRIBUTED_WORKER_COUNTS))
    ]
    output_dir = ensure_output_dir()

    print(f"[main] {len(tests)} teste(s) carregado(s)", flush=True)
    print(f"[main] output: {output_dir.resolve()}", flush=True)

    local_processes: list[mp.Process] = []
    for host, port in workers:
        p = mp.Process(target=run_worker_server, args=(host, port))
        p.start()
        local_processes.append(p)
    time.sleep(1.0)

    try:
        results = run_benchmarks(
            tests=tests,
            distributed_workers=workers,
            local_parallel_workers=LOCAL_WORKERS,
            out_dir=output_dir,
        )

        print_table(results)

        csv_path = write_csv(results, output_dir)
        plot_paths = plot_results(results, output_dir)

        print(f"\nArquivos gerados em: {output_dir.resolve()}", flush=True)
        for path in [csv_path, *plot_paths]:
            print(f"- {path.name}", flush=True)

    finally:
        shutdown_workers(workers)
        for p in local_processes:
            if p.is_alive():
                p.join(timeout=2)
            if p.is_alive():
                p.terminate()

    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
