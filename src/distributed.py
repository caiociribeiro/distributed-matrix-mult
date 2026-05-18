from __future__ import annotations

import socket
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Sequence, Tuple

import numpy as np

from src.matrix_ops import concat_vertical, split_matrix_rows
from src.socket_utils import recv_message, send_message


Matrix = np.ndarray
WorkerEndpoint = Tuple[str, int]

# Timeout apenas para estabelecer conexão TCP; após conectar, removemos o timeout
# para que operações longas (multiplicação manual de matrizes grandes) não disparem
# socket.timeout no recv_message do cliente.
CONNECT_TIMEOUT = 30  # segundos


def parse_hosts(hosts_arg: str | None, default_workers: int, base_port: int) -> List[WorkerEndpoint]:
    if hosts_arg:
        endpoints: List[WorkerEndpoint] = []
        for raw in hosts_arg.split(","):
            raw = raw.strip()
            if not raw:
                continue
            if ":" not in raw:
                raise ValueError(f"Endereço inválido em --hosts: '{raw}'. Use host:porta")
            host, port_str = raw.rsplit(":", 1)
            endpoints.append((host.strip(), int(port_str.strip())))
        if not endpoints:
            raise ValueError("Nenhum host válido informado em --hosts.")
        return endpoints

    return [("127.0.0.1", base_port + i) for i in range(default_workers)]


def _send_job(sub_a: Matrix, b: Matrix, endpoint: WorkerEndpoint, chunk_index: int) -> Matrix:
    """
    Envia um chunk para um worker e aguarda o resultado.

    Bug corrigido: socket.create_connection(timeout=X) define o timeout para
    TODAS as operações subsequentes da socket, não apenas para a conexão.
    Para matrizes grandes, o worker pode demorar muito mais que CONNECT_TIMEOUT
    calculando a multiplicação manual. Após conectar, removemos o timeout com
    sock.settimeout(None) para que o recv_message aguarde indefinidamente.
    """
    host, port = endpoint
    print(
        f"[client] enviando chunk={chunk_index} para {host}:{port} "
        f"A={sub_a.shape} B={b.shape}",
        flush=True,
    )
    request = {
        "action": "multiply",
        "chunk_index": chunk_index,
        "sub_a": sub_a,
        "b": b,
    }
    with socket.create_connection((host, port), timeout=CONNECT_TIMEOUT) as sock:
        # Remove timeout depois de conectar: recv aguarda o resultado sem limite
        sock.settimeout(None)
        send_message(sock, request)
        response = recv_message(sock)

    if not isinstance(response, dict) or not response.get("ok", False):
        raise RuntimeError(f"Falha do worker {host}:{port}: {response}")

    result = response["result"]
    print(f"[client] chunk={chunk_index} recebido de {host}:{port}", flush=True)
    return result


def distributed_multiply(a: Matrix, b: Matrix, workers: Sequence[WorkerEndpoint]) -> Matrix:
    """
    Divide A em blocos de linhas e envia cada bloco a um worker em PARALELO.

    Bug corrigido: a versão anterior enviava os jobs sequencialmente com um for
    simples, fazendo o tempo total ser a SOMA dos tempos individuais, o que é
    igual ou pior que a execução serial. Com ThreadPoolExecutor, todos os workers
    computam ao mesmo tempo e o tempo total se aproxima do MÁXIMO individual
    (speedup real ≈ N workers).
    """
    if not workers:
        raise ValueError("É necessário pelo menos um worker.")

    chunks = split_matrix_rows(a, len(workers))
    n_chunks = len(chunks)
    print(
        f"[client] distribuindo A={a.shape} em {n_chunks} chunk(s) "
        f"para {len(workers)} worker(s)",
        flush=True,
    )

    if n_chunks == 1:
        return _send_job(chunks[0], b, workers[0], 0)

    results: list[tuple[int, Matrix]] = [None] * n_chunks  # type: ignore[list-item]

    with ThreadPoolExecutor(max_workers=n_chunks) as executor:
        future_to_idx = {
            executor.submit(
                _send_job,
                chunk,
                b,
                workers[idx % len(workers)],
                idx,
            ): idx
            for idx, chunk in enumerate(chunks)
        }

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            # Propaga exceções do worker imediatamente
            results[idx] = (idx, future.result())

    results_sorted = [block for _, block in sorted(results, key=lambda x: x[0])]
    out = concat_vertical(results_sorted)
    print(f"[client] resultado distribuído montado com shape={out.shape}", flush=True)
    return out


def shutdown_workers(workers: Sequence[WorkerEndpoint]) -> None:
    for host, port in workers:
        try:
            print(f"[client] encerrando worker {host}:{port}", flush=True)
            with socket.create_connection((host, port), timeout=5) as sock:
                sock.settimeout(None)
                send_message(sock, {"action": "shutdown"})
                _ = recv_message(sock)
        except Exception as exc:
            print(f"[client] não foi possível encerrar {host}:{port}: {exc}", flush=True)
            continue
