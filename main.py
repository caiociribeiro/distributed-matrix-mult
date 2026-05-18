from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from typing import Sequence

from src.benchmark import run_benchmarks
from src.distributed import parse_hosts, shutdown_workers
from src.parser import parse_test_file
from src.report import ensure_output_dir, plot_results, print_table, write_csv, write_summary
from src.worker import run_worker_server


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulador de multiplicação de matrizes distribuída",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("tests_file", nargs="?", help="Caminho do arquivo de testes")
    parser.add_argument("--mode", choices=["local", "socket"], default="local")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--hosts", type=str, default=None)
    parser.add_argument("--base-port", type=int, default=5000)
    parser.add_argument("--low", type=int, default=-9)
    parser.add_argument("--high", type=int, default=9)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])

    print(
        f"[main] args: worker={args.worker} mode={args.mode} "
        f"workers={args.workers} tests_file={args.tests_file}",
        flush=True,
    )

    if args.worker:
        print(f"[main] iniciando worker em {args.host}:{args.port}", flush=True)
        run_worker_server(args.host, args.port)
        return 0

    if not args.tests_file:
        print("Uso: python main.py <arquivo_de_testes>", file=sys.stderr)
        return 2

    tests = parse_test_file(args.tests_file)
    workers = parse_hosts(args.hosts, args.workers, args.base_port)
    output_dir = ensure_output_dir()

    print(f"[main] {len(tests)} teste(s) carregado(s)", flush=True)
    print(f"[main] workers: {workers}", flush=True)
    print(f"[main] output: {output_dir.resolve()}", flush=True)

    local_processes: list[mp.Process] = []
    if args.mode == "local":
        for host, port in workers:
            print(f"[main] subindo worker local {host}:{port}", flush=True)
            p = mp.Process(target=run_worker_server, args=(host, port))
            p.start()
            local_processes.append(p)

        time.sleep(1.0)

    try:
        results = run_benchmarks(
            tests=tests,
            workers=workers[: max(1, min(args.workers, len(workers)))],
            low=args.low,
            high=args.high,
        )

        print_table(results)

        csv_path = write_csv(results, output_dir)
        times_path, speedup_path = plot_results(results, output_dir)
        summary_path = write_summary(results, output_dir)

        print(f"\nArquivos gerados em: {output_dir.resolve()}", flush=True)
        print(f"- {csv_path.name}", flush=True)
        print(f"- {times_path.name}", flush=True)
        print(f"- {speedup_path.name}", flush=True)
        print(f"- {summary_path.name}", flush=True)

    finally:
        if args.mode == "local":
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
