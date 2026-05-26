import socket
import threading
import multiprocessing as mp

from .matrix_ops import multiply_serial, multiply_parallel_processes
from .socket_utils import recv_message, send_message


# iniciar os workers em processos separados
def start_workers(workers):
    processes = []

    for host, port in workers:
        p = mp.Process(target=_run_worker_server, args=(host, port))
        p.start()
        processes.append(p)

    return processes


# envia sinal de shutdown para todos os workers
def shutdown_workers(workers):
    for host, port in workers:
        with socket.create_connection((host, port)) as sock:
            send_message(sock, {"action": "shutdown"})
            recv_message(sock)


# worker responsavel por receber as tarefas de multiplicacao, processa-las e enviar os resultados de volta para o coordenador
# se serial, chama multiply_serial diretamente, simulando o uso de 1 core por worker
# se paralelo, chama multiply_parallel_processes com o numero de workers internos configurado, simulando o uso de mais de 1 core por worker
def _handle_client(conn):
    with conn:
        request = recv_message(conn)

        if request["action"] == "shutdown":
            send_message(conn, {"ok": True})
            return

        sub_a = request["sub_a"]
        b = request["b"]

        if request["parallel"]:
            result = multiply_parallel_processes(
                sub_a,
                b,
                workers=request["internal_workers"],
            )
        else:
            result = multiply_serial(sub_a, b)

        send_message(
            conn,
            {
                "ok": True,
                "chunk_index": request["chunk_index"],
                "result": result,
            },
        )


# inicia o server do worker via socket TCP
def _run_worker_server(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind((host, port))
        server.listen()

        while True:
            conn, _ = server.accept()

            thread = threading.Thread(
                target=_handle_client,
                args=(conn,),
                daemon=True,
            )

            thread.start()
