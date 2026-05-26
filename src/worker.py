import socket
import threading
import multiprocessing as mp

from .matrix_ops import multiply_serial, multiply_parallel_processes
from .socket_utils import recv_message, send_message


# iniciar os workers em processos separados
def start_workers(workers):
    processes = []

    print(f"[workers] iniciando {len(workers)} worker(s)")

    for host, port in workers:
        p = mp.Process(target=_run_worker_server, args=(host, port))
        p.start()

        print(f"[workers] worker iniciado em {host}:{port}")

        processes.append(p)

    return processes


# envia sinal de shutdown para todos os workers
def shutdown_workers(workers):
    print("[workers] encerrando workers")

    for host, port in workers:
        print(f"[workers] enviando shutdown para {host}:{port}")

        with socket.create_connection((host, port)) as sock:
            send_message(sock, {"action": "shutdown"})
            recv_message(sock)


# worker responsavel por receber as tarefas de multiplicacao
def _handle_client(conn, port):
    with conn:
        print(f"[worker {port}] conexão recebida")

        request = recv_message(conn)

        if request["action"] == "shutdown":
            print(f"[worker {port}] encerrando")

            send_message(conn, {"ok": True})
            return

        sub_a = request["sub_a"]
        b = request["b"]

        print(f"[worker {port}] processando chunk {sub_a.shape[0]}x{sub_a.shape[1]}")

        if request["parallel"]:
            print(
                f"[worker {port}] usando "
                f"{request['internal_workers']} processos internos"
            )

            result = multiply_parallel_processes(
                sub_a,
                b,
                workers=request["internal_workers"],
            )

        else:
            result = multiply_serial(sub_a, b)

        print(f"[worker {port}] chunk finalizado")

        send_message(
            conn,
            {
                "ok": True,
                "result": result,
            },
        )


# inicia o server do worker via socket TCP
def _run_worker_server(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind((host, port))
        server.listen()

        print(f"[worker {port}] aguardando conexões")

        while True:
            conn, _ = server.accept()

            thread = threading.Thread(
                target=_handle_client,
                args=(conn, port),
                daemon=True,
            )

            thread.start()
