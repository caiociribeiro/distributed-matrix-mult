from __future__ import annotations

import numpy as np


Matrix = np.ndarray


def generate_matrix(rows: int, cols: int, low: int = -9, high: int = 9) -> Matrix:
    return np.random.randint(low, high + 1, size=(rows, cols), dtype=np.int64)


def multiply_serial(a: Matrix, b: Matrix) -> Matrix:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("As matrizes devem ser bidimensionais.")
    if a.shape[1] != b.shape[0]:
        raise ValueError("Dimensões incompatíveis para multiplicação.")

    rows_a, cols_a = a.shape
    _, cols_b = b.shape
    result = np.zeros((rows_a, cols_b), dtype=np.int64)

    for i in range(rows_a):
        for j in range(cols_b):
            total = 0
            for k in range(cols_a):
                total += int(a[i, k]) * int(b[k, j])
            result[i, j] = total

    return result


def multiply_block_rows(a_block: Matrix, b: Matrix) -> Matrix:
    return multiply_serial(a_block, b)


def split_matrix_rows(matrix: Matrix, num_parts: int):
    if num_parts <= 1:
        return [matrix]
    if matrix.size == 0:
        return [matrix]

    num_parts = min(num_parts, matrix.shape[0])
    return [chunk for chunk in np.array_split(matrix, num_parts, axis=0) if chunk.size > 0]


def concat_vertical(blocks):
    if not blocks:
        return np.empty((0, 0), dtype=np.int64)
    if len(blocks) == 1:
        return blocks[0]
    return np.vstack(blocks)


def matrices_equal(a: Matrix, b: Matrix) -> bool:
    return np.array_equal(a, b)
