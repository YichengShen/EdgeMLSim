import pickle
import struct
import socket
import sys
import threading
import time
from Msg import *
from mxnet import nd
import numpy as np

# Global Var for ease of testing
SERVER_PORT = 9999
EDGE_PORT = 6666

def server_handle_connection(host, port, instance, persistent_connection):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    #Bind socket to local host and port
    try:
        s.bind((host, port))
    except socket.error as msg:
        print('Bind failed. Error :', msg)
        sys.exit()
    #Start listening on socket
    s.listen()
    print('Socket now listening')
    s.settimeout(10)

    while True:
        try:
            conn, addr = s.accept()
            print('Connected with ' + addr[0] + ':' + str(addr[1]))
            threading.Thread(target=connection_thread, args=(conn, instance, persistent_connection)).start()
            with instance.cv:
                instance.connections.append(conn)
                send_message(conn, instance.type, PayloadType.PARAMETER, instance.parameter)
                instance.cv.notify()
        except:
            if instance.terminated:
                break 

    # Close all exisiting connections
    for conn in instance.connections:
        conn.close()

    print('Connection loop exit')

    s.close()

def connection_thread(conn, instance, persistent_connection):
    while not instance.terminated:
        msg = wait_for_message(conn)
        if msg:
            with instance.cv:
                instance.accumulative_gradients.append(msg.payload)
                instance.cv.notify()
        if not persistent_connection:
            break

def client_build_connection(host, port):
    #create an INET, STREAMing socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    except socket.error:
        print('Failed to create socket')
        sys.exit()

    try:
        remote_ip = socket.gethostbyname(host)

    except socket.gaierror:
        #could not resolve
        print('Hostname could not be resolved. Exiting')
        sys.exit()

    #Connect to remote server
    s.connect((remote_ip, port))
    #Wait for parameters from remote server
    params = wait_for_message(s)
    return s, params

def send_message(conn, source_type, payload_type, payload):
    msg = Msg(source_type, payload_type, payload)
    data = pickle.dumps(msg)

    # Add length of data as prefix to the msg
    s = struct.pack('>I', len(data)) + data
    conn.sendall(s)


def wait_for_message(conn):
    # Retreat the length of the data (The 4-bytes prefix)
    msglen = wait_for_message_helper(conn, 4)
    if not msglen:
        return None
    msglen = struct.unpack('>I', msglen[0])[0]

    # Retreat data using its length
    data = wait_for_message_helper(conn, msglen)
    return pickle.loads(b"".join(data))

def wait_for_message_helper(conn, n):
    data = []
    # Retreat data of length n
    while len(b"".join(data)) < n:
        packet = conn.recv(n - len(b"".join(data)))
        if not packet:
            return None
        data.append(packet)
    return data
    # while data_len < n:
    #     packet = conn.recv(n - data_len)
    #     if not packet:
    #         return None
    #     data.append(packet)
    #     data_len += len(b"".join(data))
    # return data