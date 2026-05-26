from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np

Matrix = np.ndarray


# retorna uma matriz de inteiros eleatorios
def generate_matrix(rows, cols, low, high):
    return np.random.randint(low, high + 1, size=(rows, cols), dtype=np.int64)


# multiplicacao de matrizes utilizando 3 loops aninhados
# retorna o resultado da multiplicação
def multiply_serial(a, b):
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


# divide a matriz em chunks
def split_matrix_rows(matrix, num_parts):
    if num_parts <= 1:
        return [matrix]
    if matrix.size == 0:
        return [matrix]

    num_parts = min(num_parts, matrix.shape[0])
    return [
        chunk for chunk in np.array_split(matrix, num_parts, axis=0) if chunk.size > 0
    ]


# junta os blocos de volta em uma matriz completa
def concat(blocks):
    return np.vstack(blocks)


# compara se duas matrizes sao iguais
def matrices_equal(a, b):
    return np.array_equal(a, b)


# estima a carga de trabalho da multiplicacao de matrizes (quantidade de multiplicacoes e somas)
def estimate_work_units(a, b):
    return int(a.shape[0] * a.shape[1] * b.shape[1])


# faz a multiplicacao de um chunk da matriz A com a matriz B
def _process_chunk(args):
    a_block, b = args
    return multiply_serial(a_block, b)


# multiplicacao utilizando threads
def multiply_parallel_threads(a, b, workers):
    if workers <= 1:
        return multiply_serial(a, b)

    chunks = split_matrix_rows(a, workers)
    if len(chunks) == 1:
        return multiply_serial(a, b)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(_process_chunk, [(chunk, b) for chunk in chunks]))
    return concat(results)


# multiplicacao utilizando processos
def multiply_parallel_processes(a, b, workers):
    if workers <= 1:
        return multiply_serial(a, b)

    chunks = split_matrix_rows(a, workers)
    if len(chunks) == 1:
        return multiply_serial(a, b)

    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(_process_chunk, [(chunk, b) for chunk in chunks]))
    return concat(results)
