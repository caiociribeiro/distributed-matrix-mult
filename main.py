from __future__ import annotations

import sys
import time
from pathlib import Path

from src.benchmark import run_benchmarks
from src.parser import parse_test_file
from src.report import (
    plot_results,
    print_table,
    write_csv,
)
from src.worker import start_workers, shutdown_workers


BASE_PORT = 5000
DISTRIBUTED_WORKER_COUNTS = [2, 4, 6]
LOCAL_WORKERS = 4


def main():
    if len(sys.argv) < 2:
        print("Uso: python main.py <arquivo_de_testes>")
        return

    tests = parse_test_file(sys.argv[1])

    # cria lista de workers locais
    workers = [
        ("127.0.0.1", BASE_PORT + i) for i in range(max(DISTRIBUTED_WORKER_COUNTS))
    ]

    # cria diretorio de output
    output_dir = Path("results") / time.strftime("%Y%m%d%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "matrices").mkdir(parents=True, exist_ok=True)

    print(f"[main] {len(tests)} teste(s) carregado(s)")
    print(f"[main] output: {output_dir.resolve()}")

    # inicia os workers
    local_processes = start_workers(workers)

    # espera os sockets iniciarem
    time.sleep(1)

    try:
        results = run_benchmarks(
            tests=tests,
            distributed_workers=workers,
            distributed_worker_counts=DISTRIBUTED_WORKER_COUNTS,
            local_parallel_workers=LOCAL_WORKERS,
            out_dir=output_dir,
        )

        print_table(results)

        csv_path = write_csv(results, output_dir)
        plot_paths = plot_results(results, output_dir)

        print(f"\nArquivos gerados em: {output_dir.resolve()}")

        for path in [csv_path, *plot_paths]:
            print(f"- {path.name}")

    finally:
        # encerra workers via socket
        shutdown_workers(workers)

        # encerra processos locais
        for p in local_processes:
            p.join(timeout=2)

            if p.is_alive():
                p.terminate()


if __name__ == "__main__":
    raise SystemExit(main())
