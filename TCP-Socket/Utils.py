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
SIM_PORT_WORKER = 10002
SIM_PORT_CLOUD = SIM_PORT_WORKER + 10000
SIM_PORT_EDGE = SIM_PORT_CLOUD + 10000

def server_handle_connection(host, port, instance, persistent_connection, source_type=None, client_type=None):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    #Bind socket to local host and port
    try:
        s.bind((host, port))
        instance.port = s.getsockname()[1]
        # print(s.getsockname()[1])
    except socket.error as msg:
        print('Bind failed. Error :', msg)
        sys.exit()
    #Start listening on socket
    s.listen()
    # print(instance.type, 'now listening')
    s.settimeout(10)

    while True:
        try:
            conn, addr = s.accept()
            print('Connected with ' + addr[0] + ':' + str(addr[1]))
            if source_type == None:
                threading.Thread(target=connection_thread, args=(conn, instance, persistent_connection, source_type)).start()
            with instance.cv:
                if source_type == InstanceType.SIMULATOR:
                    if client_type == InstanceType.WORKER:
                        instance.worker_conns.append(conn)
                        # assign id to worker
                        instance.worker_id_free.append(instance.worker_count) 
                        # send id to worker
                        send_message(conn, instance.type, PayloadType.ID, instance.worker_count)
                        instance.worker_count += 1
                    elif client_type == InstanceType.CLOUD_SERVER:
                        instance.cloud_conn = conn
                    elif client_type == InstanceType.EDGE_SERVER:
                        instance.edge_conns.append(conn)
                else:
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

def connection_thread(conn, instance, persistent_connection, source_type):
    while not instance.terminated:
        try:
            msg = wait_for_message(conn)
        except OSError:
            sys.exit()
        if msg:
            if source_type == None:
                if msg.get_payload_type() == PayloadType.GRADIENT:
                    # used for both Cloud and Edge
                    with instance.cv:     
                        instance.accumulative_gradients.append(msg.payload)
                        instance.cv.notify()
                elif msg.get_payload_type() == PayloadType.REQUEST:
                    # used for Edge Server (when workers ask for parameters)
                    send_message(conn, instance.type, PayloadType.PARAMETER, instance.parameter)
        if not persistent_connection:
            break

def client_build_connection(host, port, wait_initial_msg=True):
    # create an INET, STREAMing socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    except socket.error:
        print('Failed to create socket')
        sys.exit()

    try:
        remote_ip = socket.gethostbyname(host)

    except socket.gaierror:
        # could not resolve
        print('Hostname could not be resolved. Exiting')
        sys.exit()

    # Connect to remote server
    s.connect((remote_ip, port))

    if wait_initial_msg:
        # Wait for messages from remote server
        msg = wait_for_message(s)
        return s, msg
    else:
        return s

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