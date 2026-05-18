from __future__ import annotations

import socket
import threading
from typing import Tuple

from src.matrix_ops import multiply_block_rows
from src.socket_utils import recv_message, send_message


def handle_client(conn: socket.socket, addr: Tuple[str, int]) -> None:
    with conn:
        try:
            print(f"[worker] conexão recebida de {addr}", flush=True)
            request = recv_message(conn)
            if not isinstance(request, dict):
                raise ValueError("Mensagem inválida.")

            action = request.get("action")
            print(f"[worker] ação={action} de {addr}", flush=True)

            if action == "shutdown":
                send_message(conn, {"ok": True, "shutdown": True})
                print(f"[worker] shutdown enviado para {addr}", flush=True)
                return

            if action != "multiply":
                raise ValueError(f"Ação desconhecida: {action}")

            sub_a = request["sub_a"]
            b = request["b"]
            chunk_index = int(request["chunk_index"])

            print(
                f"[worker] multiplicando chunk={chunk_index} "
                f"A={sub_a.shape} B={b.shape}",
                flush=True,
            )

            result = multiply_block_rows(sub_a, b)

            send_message(
                conn,
                {
                    "ok": True,
                    "chunk_index": chunk_index,
                    "result": result,
                },
            )
            print(f"[worker] resultado enviado chunk={chunk_index}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[worker] erro: {exc}", flush=True)
            try:
                send_message(conn, {"ok": False, "error": str(exc)})
            except Exception:
                pass


def run_worker_server(host: str, port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen()
        print(f"[worker] escutando em {host}:{port}", flush=True)

        while True:
            conn, addr = server.accept()
            thread = threading.Thread(target=handle_client, args=(conn, addr), daemon=True)
            thread.start()
