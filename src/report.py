from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .matrix_ops import Matrix


MODE_LABELS = {
    "serial": "Serial",
    "threads": "Threads (4)",
    "processes": "Processos (4)",
    "distributed_serial": "Distribuída (1 core)",
    "distributed_parallel": "Distribuída (>1 core)",
}


# salva uma matriz em csv
def save_matrix_csv(path, matrix):
    np.savetxt(path, matrix, fmt="%d", delimiter=",")


# salva os artefatos de um teste (matrizes A, B e os resultados de cada modo)
def save_test_artifacts(
    out_dir: Path,
    test_index: int,
    a: Matrix,
    b: Matrix,
    outputs_by_mode: dict[str, Matrix],
):
    test_dir = out_dir / "matrices" / f"test_{test_index:03d}"
    test_dir.mkdir(parents=True, exist_ok=True)

    save_matrix_csv(test_dir / "A.csv", a)
    save_matrix_csv(test_dir / "B.csv", b)

    for mode, matrix in outputs_by_mode.items():
        save_matrix_csv(test_dir / f"{mode}.csv", matrix)

    return test_dir


def write_csv(results, out_dir):
    path = out_dir / "results.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "test_index",
                "mode",
                "workers",
                "rows_a",
                "cols_a",
                "rows_b",
                "cols_b",
                "work_units",
                "elapsed_ms",
                "speedup",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.test_index,
                    r.mode,
                    r.workers,
                    r.rows_a,
                    r.cols_a,
                    r.rows_b,
                    r.cols_b,
                    r.work_units,
                    f"{r.elapsed_ms:.6f}",
                    f"{r.speedup:.6f}",
                ]
            )
    return path


def print_table(results):
    headers = [
        "Teste",
        "Modo",
        "Workers",
        "A",
        "B",
        "Work",
        "Tempo (ms)",
        "Speedup",
    ]
    rows = [
        [
            str(r.test_index),
            MODE_LABELS.get(r.mode, r.mode),
            str(r.workers),
            f"{r.rows_a}x{r.cols_a}",
            f"{r.rows_b}x{r.cols_b}",
            str(r.work_units),
            f"{r.elapsed_ms:.2f}",
            f"{r.speedup:.2f}",
        ]
        for r in results
    ]

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt(row):
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))

    sep = "-+-".join("-" * w for w in widths)
    print(fmt(headers))
    print(sep)
    for row in rows:
        print(fmt(row))


def plot_results(results, out_dir):
    generated: list[Path] = []

    target_workers = [2, 4, 6]
    modes = [
        "serial",
        "threads",
        "processes",
        "distributed_serial",
        "distributed_parallel",
    ]

    for workers in target_workers:
        fig1 = plt.figure(figsize=(13, 7))

        for mode in modes:
            if mode == "serial":
                mode_subset = [r for r in results if r.mode == mode]
            elif mode in {"threads", "processes"}:
                mode_subset = [r for r in results if r.mode == mode and r.workers == 4]
            else:
                mode_subset = [
                    r for r in results if r.mode == mode and r.workers == workers
                ]

            if not mode_subset:
                continue

            mode_subset = sorted(mode_subset, key=lambda r: r.test_index)
            x_labels = [r.signature for r in mode_subset]
            y_time = [r.elapsed_ms for r in mode_subset]

            plt.plot(
                x_labels,
                y_time,
                marker="o",
                linewidth=2,
                label=MODE_LABELS.get(mode, mode),
            )

        plt.xlabel("Dimensões das matrizes")
        plt.ylabel("Tempo (ms)")
        plt.title(f"Tempo por tamanho da instância — workers distribuídos = {workers}")
        plt.grid(True, linestyle="--", alpha=0.35)
        plt.xticks(rotation=15, ha="right")
        plt.legend()
        plt.tight_layout()

        times_path = out_dir / f"times_workers_{workers}.png"
        fig1.savefig(times_path, dpi=160)
        plt.close(fig1)
        generated.append(times_path)

        fig2 = plt.figure(figsize=(13, 7))

        for mode in modes:
            if mode == "serial":
                mode_subset = [r for r in results if r.mode == mode]
            elif mode in {"threads", "processes"}:
                mode_subset = [r for r in results if r.mode == mode and r.workers == 4]
            else:
                mode_subset = [
                    r for r in results if r.mode == mode and r.workers == workers
                ]

            if not mode_subset:
                continue

            mode_subset = sorted(mode_subset, key=lambda r: r.test_index)
            x_labels = [r.signature for r in mode_subset]
            y_speedup = [r.speedup for r in mode_subset]

            plt.plot(
                x_labels,
                y_speedup,
                marker="o",
                linewidth=2,
                label=MODE_LABELS.get(mode, mode),
            )

        plt.xlabel("Dimensões das matrizes")
        plt.ylabel("Speedup sobre o serial")
        plt.title(
            f"Speedup por tamanho da instância — workers distribuídos = {workers}"
        )
        plt.grid(True, linestyle="--", alpha=0.35)
        plt.xticks(rotation=15, ha="right")
        plt.legend()
        plt.tight_layout()

        speedup_path = out_dir / f"speedup_workers_{workers}.png"
        fig2.savefig(speedup_path, dpi=160)
        plt.close(fig2)
        generated.append(speedup_path)

    return generated
