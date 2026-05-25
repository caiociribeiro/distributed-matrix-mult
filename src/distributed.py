from __future__ import annotations

import os
import socket
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Sequence, Tuple

import numpy as np

from .matrix_ops import concat_vertical, split_matrix_rows
from .socket_utils import recv_message, send_message


Matrix = np.ndarray
WorkerEndpoint = Tuple[str, int]
CONNECT_TIMEOUT = 30


def parse_hosts(
    hosts_arg: str | None, default_workers: int, base_port: int
) -> List[WorkerEndpoint]:
    if hosts_arg:
        endpoints: List[WorkerEndpoint] = []
        for raw in hosts_arg.split(","):
            raw = raw.strip()
            if not raw:
                continue
            if ":" not in raw:
                raise ValueError(
                    f"Endereço inválido em --hosts: '{raw}'. Use host:porta"
                )
            host, port_str = raw.rsplit(":", 1)
            endpoints.append((host.strip(), int(port_str.strip())))
        if not endpoints:
            raise ValueError("Nenhum host válido informado em --hosts.")
        return endpoints

    return [("127.0.0.1", base_port + i) for i in range(default_workers)]


def _send_job(
    sub_a: Matrix,
    b: Matrix,
    endpoint: WorkerEndpoint,
    chunk_index: int,
    *,
    parallel: bool,
    internal_workers: int,
) -> Matrix:
    host, port = endpoint
    request = {
        "action": "multiply",
        "chunk_index": chunk_index,
        "sub_a": sub_a,
        "b": b,
        "parallel": parallel,
        "internal_workers": internal_workers,
    }
    with socket.create_connection((host, port), timeout=CONNECT_TIMEOUT) as sock:
        sock.settimeout(None)
        send_message(sock, request)
        response = recv_message(sock)

    if not isinstance(response, dict) or not response.get("ok", False):
        raise RuntimeError(f"Falha do worker {host}:{port}: {response}")

    return response["result"]


def _distributed_multiply(
    a: Matrix,
    b: Matrix,
    workers: Sequence[WorkerEndpoint],
    *,
    parallel: bool,
    internal_workers: int,
) -> Matrix:
    if not workers:
        raise ValueError("É necessário pelo menos um worker.")

    chunks = split_matrix_rows(a, len(workers))

    results: list[tuple[int, Matrix] | None] = [None] * len(chunks)

    with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
        future_to_idx = {
            executor.submit(
                _send_job,
                chunk,
                b,
                workers[idx % len(workers)],
                idx,
                parallel=parallel,
                internal_workers=internal_workers,
            ): idx
            for idx, chunk in enumerate(chunks)
        }

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = (idx, future.result())

    ordered = [block for _, block in sorted(results, key=lambda item: item[0])]

    return concat_vertical(ordered)


def distributed_multiply_serial(
    a: Matrix, b: Matrix, workers: Sequence[WorkerEndpoint]
) -> Matrix:
    return _distributed_multiply(a, b, workers, parallel=False, internal_workers=1)


def distributed_multiply_parallel(
    a: Matrix,
    b: Matrix,
    workers: Sequence[WorkerEndpoint],
    internal_workers: int | None = None,
) -> Matrix:
    if internal_workers is None:
        cpu = os.cpu_count() or 2
        internal_workers = max(2, cpu // max(1, len(workers)))
    return _distributed_multiply(
        a, b, workers, parallel=True, internal_workers=internal_workers
    )


def shutdown_workers(workers: Sequence[WorkerEndpoint]) -> None:
    for host, port in workers:
        try:
            with socket.create_connection((host, port), timeout=5) as sock:
                sock.settimeout(None)
                send_message(sock, {"action": "shutdown"})
                _ = recv_message(sock)
        except Exception:
            continue
