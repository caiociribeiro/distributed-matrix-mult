from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Sequence, Tuple

import matplotlib.pyplot as plt

from src.models import BenchmarkResult


def ensure_output_dir() -> Path:
    out_dir = Path("results") / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def write_csv(results: Sequence[BenchmarkResult], out_dir: Path) -> Path:
    path = out_dir / "results.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "test_index",
                "shape_a",
                "shape_b",
                "workers",
                "serial_ms",
                "distributed_ms",
                "speedup",
                "correct",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.test_index,
                    r.a_shape,
                    r.b_shape,
                    r.workers,
                    f"{r.serial_ms:.6f}",
                    f"{r.distributed_ms:.6f}",
                    f"{r.speedup:.6f}",
                    r.correct,
                ]
            )
    return path


def print_table(results: Sequence[BenchmarkResult]) -> None:
    headers = ["Teste", "A", "B", "Workers", "Serial (ms)", "Distrib. (ms)", "Speedup", "OK"]
    rows = [
        [
            str(r.test_index),
            r.a_shape,
            r.b_shape,
            str(r.workers),
            f"{r.serial_ms:.2f}",
            f"{r.distributed_ms:.2f}",
            f"{r.speedup:.2f}",
            "sim" if r.correct else "não",
        ]
        for r in results
    ]

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt(row: Sequence[str]) -> str:
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))

    sep = "-+-".join("-" * w for w in widths)
    print(fmt(headers))
    print(sep)
    for row in rows:
        print(fmt(row))


def plot_results(results: Sequence[BenchmarkResult], out_dir: Path) -> Tuple[Path, Path]:
    indices = [r.test_index for r in results]
    serial = [r.serial_ms for r in results]
    distributed = [r.distributed_ms for r in results]
    speedup = [r.speedup for r in results]

    fig1 = plt.figure(figsize=(10, 5))
    plt.plot(indices, serial, marker="o", linewidth=2, label="Serial")
    plt.plot(indices, distributed, marker="o", linewidth=2, label="Distribuída")
    plt.yscale("log")
    plt.xticks(indices)
    plt.xlabel("Teste")
    plt.ylabel("Tempo (ms)")
    plt.title("Comparação de tempos: serial vs distribuída")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    times_path = out_dir / "times_comparison.png"
    fig1.savefig(times_path, dpi=150)
    plt.close(fig1)

    fig2 = plt.figure(figsize=(10, 5))
    plt.plot(indices, speedup, marker="o", linewidth=2)
    plt.xticks(indices)
    plt.xlabel("Teste")
    plt.ylabel("Speedup (serial / distribuída)")
    plt.title("Speedup por teste")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    speedup_path = out_dir / "speedup.png"
    fig2.savefig(speedup_path, dpi=150)
    plt.close(fig2)

    return times_path, speedup_path


def write_summary(results: Sequence[BenchmarkResult], out_dir: Path) -> Path:
    path = out_dir / "summary.txt"
    with path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(
                f"Teste {r.test_index}: A={r.a_shape}, B={r.b_shape}, workers={r.workers}, "
                f"serial={r.serial_ms:.6f}ms, distribuída={r.distributed_ms:.6f}ms, "
                f"speedup={r.speedup:.6f}, correto={r.correct}\n"
            )
    return path
