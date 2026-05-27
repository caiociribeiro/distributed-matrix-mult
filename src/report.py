import csv

import matplotlib.pyplot as plt
import numpy as np


MODE_LABELS = {
    "serial": "Serial",
    "threads": "Threads",
    "processes": "Processos",
    "distributed_serial": "Distribuída (1 core)",
    "distributed_parallel": "Distribuída (>1 core)",
}


# salva uma matriz em csv
def save_matrix_csv(path, matrix):
    np.savetxt(path, matrix, fmt="%d", delimiter=",")


# salva matrizes de entrada e resultados
def save_test_artifacts(
    out_dir,
    test_index,
    a,
    b,
    outputs_by_mode,
):
    test_dir = out_dir / "matrices" / f"test_{test_index:03d}"
    test_dir.mkdir(parents=True, exist_ok=True)

    save_matrix_csv(test_dir / "A.csv", a)
    save_matrix_csv(test_dir / "B.csv", b)

    for mode, matrix in outputs_by_mode.items():
        save_matrix_csv(test_dir / f"{mode}.csv", matrix)

    return test_dir


# salva resultados em csv
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


# imprime tabela no terminal
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

    rows = []

    for r in results:
        rows.append(
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
        )

    widths = [len(h) for h in headers]

    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def format_row(row):
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))

    separator = "-+-".join("-" * w for w in widths)

    print(format_row(headers))
    print(separator)

    for row in rows:
        print(format_row(row))


# filtra resultados de um modo especifico
def get_mode_results(results, mode, workers):
    if mode in {"serial", "threads", "processes"}:
        return [r for r in results if r.mode == mode]

    return [r for r in results if r.mode == mode and r.workers == workers]


# gera grafico
def _generate_plot(
    results,
    workers,
    ylabel,
    title,
    filename,
    value_getter,
    out_dir,
):
    modes = [
        "serial",
        "threads",
        "processes",
        "distributed_serial",
        "distributed_parallel",
    ]

    fig = plt.figure(figsize=(13, 7))

    for mode in modes:
        mode_results = get_mode_results(results, mode, workers)

        if not mode_results:
            continue

        mode_results.sort(key=lambda r: r.test_index)

        x = [r.signature for r in mode_results]
        y = [value_getter(r) for r in mode_results]

        plt.plot(
            x,
            y,
            marker="o",
            linewidth=2,
            label=MODE_LABELS.get(mode, mode),
        )

    plt.xlabel("Dimensões das matrizes")
    plt.ylabel(ylabel)
    plt.title(title)

    plt.grid(True, linestyle="--", alpha=0.35)
    plt.xticks(rotation=15, ha="right")

    plt.legend()
    plt.tight_layout()

    path = out_dir / filename

    fig.savefig(path, dpi=160)
    plt.close(fig)

    return path


# gera todos os graficos
def plot_results(results, out_dir):
    generated = []

    for workers in [2, 4, 6]:
        generated.append(
            _generate_plot(
                results,
                workers,
                ylabel="Tempo (ms)",
                title=f"Tempo por tamanho da instância — workers distribuídos = {workers}",
                filename=f"times_workers_{workers}.png",
                value_getter=lambda r: r.elapsed_ms,
                out_dir=out_dir,
            )
        )

        generated.append(
            _generate_plot(
                results,
                workers,
                ylabel="Speedup sobre o serial",
                title=f"Speedup por tamanho da instância — workers distribuídos = {workers}",
                filename=f"speedup_workers_{workers}.png",
                value_getter=lambda r: r.speedup,
                out_dir=out_dir,
            )
        )

    return generated
