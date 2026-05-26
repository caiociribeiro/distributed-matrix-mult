import socket
import os
from concurrent.futures import ThreadPoolExecutor

from .matrix_ops import concat, split_matrix_rows
from .socket_utils import recv_message, send_message


# envia um chunk para um worker
def _send_job(
    sub_a,
    b,
    endpoint,
    parallel,
    internal_workers,
):
    host, port = endpoint

    print(
        f"[distributed] enviando chunk "
        f"{sub_a.shape[0]}x{sub_a.shape[1]} "
        f"para {host}:{port}"
    )

    request = {
        "action": "multiply",
        "sub_a": sub_a,
        "b": b,
        "parallel": parallel,
        "internal_workers": internal_workers,
    }

    with socket.create_connection((host, port)) as sock:
        send_message(sock, request)

        response = recv_message(sock)

    print(f"[distributed] resultado recebido de {host}:{port}")

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

    print(f"[distributed] dividindo matriz em {len(chunks)} chunk(s)")

    with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
        futures = [
            executor.submit(
                _send_job,
                chunk,
                b,
                workers[idx],
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
):
    cpu_count = os.cpu_count() or 2

    internal_workers = max(1, cpu_count // len(workers))

    return _distributed_multiply(
        a,
        b,
        workers,
        parallel=True,
        internal_workers=internal_workers,
    )
