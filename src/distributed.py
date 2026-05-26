from __future__ import annotations

import os
import socket
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Sequence, Tuple

import numpy as np

from .matrix_ops import concat, split_matrix_rows
from .socket_utils import recv_message, send_message


Matrix = np.ndarray
WorkerEndpoint = Tuple[str, int]
CONNECT_TIMEOUT = 30


def _send_job(
    sub_a,
    b,
    endpoint,
    chunk_index,
    parallel,
    internal_workers,
):
    host, port = endpoint

    request = {
        "action": "multiply",
        "chunk_index": chunk_index,
        "sub_a": sub_a,
        "b": b,
        "parallel": parallel,
        "internal_workers": internal_workers,
    }

    with socket.create_connection((host, port)) as sock:
        send_message(sock, request)
        response = recv_message(sock)

    return response["result"]


# divide a matriz em chunks e distribui para os workers
def _distributed_multiply(
    a,
    b,
    workers,
    parallel,
    internal_workers,
):
    chunks = split_matrix_rows(a, len(workers))

    with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
        futures = [
            executor.submit(
                _send_job,
                chunk,
                b,
                workers[idx],
                idx,
                parallel,
                internal_workers,
            )
            for idx, chunk in enumerate(chunks)
        ]

        results = [future.result() for future in futures]

    return concat(results)


# distribuida serial
def distributed_multiply_serial(a, b, workers):
    return _distributed_multiply(
        a,
        b,
        workers,
        parallel=False,
        internal_workers=1,
    )


# distribuida paralela
def distributed_multiply_parallel(
    a,
    b,
    workers,
    internal_workers,
):
    return _distributed_multiply(
        a,
        b,
        workers,
        parallel=True,
        internal_workers=internal_workers,
    )


def shutdown_workers(workers):
    for host, port in workers:
        with socket.create_connection((host, port)) as sock:
            send_message(sock, {"action": "shutdown"})
            recv_message(sock)
