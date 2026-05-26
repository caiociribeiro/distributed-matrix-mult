import pickle
import socket
import struct


HEADER_SIZE = 8
RECV_BUFFER = 1 << 20


def send_message(sock: socket.socket, payload):
    data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack("!Q", len(data)))
    sock.sendall(data)


def recv_exact(sock: socket.socket, nbytes: int):
    buf = bytearray(nbytes)
    view = memoryview(buf)
    received = 0
    while received < nbytes:
        chunk = min(nbytes - received, RECV_BUFFER)
        n = sock.recv_into(view[received : received + chunk], chunk)
        if n == 0:
            raise ConnectionError("Conexão encerrada antes do recebimento completo.")
        received += n
    return bytes(buf)


def recv_message(sock: socket.socket):
    size = struct.unpack("!Q", recv_exact(sock, HEADER_SIZE))[0]
    payload = recv_exact(sock, size)
    return pickle.loads(payload)
