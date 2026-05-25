from __future__ import annotations

import socket
import threading
from typing import Tuple

from .matrix_ops import multiply_block_rows, multiply_parallel_processes
from .socket_utils import recv_message, send_message


def handle_client(conn: socket.socket, addr: Tuple[str, int]) -> None:
    with conn:
        try:
            request = recv_message(conn)
            if not isinstance(request, dict):
                raise ValueError("Mensagem inválida.")

            action = request.get("action")

            if action == "shutdown":
                send_message(conn, {"ok": True, "shutdown": True})
                return

            if action != "multiply":
                raise ValueError(f"Ação desconhecida: {action}")

            sub_a = request["sub_a"]
            b = request["b"]
            chunk_index = int(request["chunk_index"])
            parallel = bool(request.get("parallel", False))
            internal_workers = int(request.get("internal_workers", 2))

            if parallel:
                result = multiply_parallel_processes(sub_a, b, workers=max(2, internal_workers))
            else:
                result = multiply_block_rows(sub_a, b)

            send_message(
                conn,
                {
                    "ok": True,
                    "chunk_index": chunk_index,
                    "result": result,
                },
            )
        except Exception as exc:  # noqa: BLE001
            try:
                send_message(conn, {"ok": False, "error": str(exc)})
            except Exception:
                pass


def run_worker_server(host: str, port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen()

        while True:
            conn, addr = server.accept()
            thread = threading.Thread(target=handle_client, args=(conn, addr), daemon=True)
            thread.start()
