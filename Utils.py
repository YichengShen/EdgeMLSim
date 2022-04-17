import pickle
import struct
import socket
import sys
import threading
import time
from Msg import *
from mxnet import nd
import numpy as np
import yaml


CFG = yaml.load(open('config/config.yml', 'r'), Loader=yaml.FullLoader)


def server_handle_connection(host, port, instance, persistent_connection, source_type=None, client_type=None):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # the SO_REUSEADDR flag tells the kernel to reuse a local socket in TIME_WAIT state,
    # without waiting for its natural timeout to expire
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    # Bind socket to local host and port
    try:
        s.bind((host, port))
    except socket.error as msg:
        print('Bind failed. Error :', msg)
        sys.exit()
    # Start listening on socket
    s.listen()
    # print(instance.type, 'now listening')
    s.settimeout(10)

    while True:
        try:
            conn, addr = s.accept()
            print('Connected with ' + addr[0] + ':' + str(addr[1]))
            if source_type == None or source_type == InstanceType.EDGE_SERVER:
                threading.Thread(target=connection_thread, args=(
                    conn, instance, persistent_connection, source_type)).start()
            with instance.cv:
                if source_type == InstanceType.SIMULATOR:
                    if client_type == InstanceType.WORKER:
                        instance.worker_conns.append(conn)
                        # TODO: revisit worker ID
                        # assign id to worker
                        instance.worker_id_free.add(instance.worker_count)
                        # send id to worker
                        send_message(conn, instance.type,
                                     PayloadType.ID, instance.worker_count)
                        instance.worker_count += 1
                    elif client_type == InstanceType.CLOUD_SERVER:
                        instance.cloud_conn = conn
                    elif client_type == InstanceType.EDGE_SERVER:
                        instance.edge_conns.append(conn)
                else:
                    instance.connections.append(conn)
                    if source_type != InstanceType.EDGE_SERVER:
                        send_message(conn, instance.type,
                                     PayloadType.PARAMETER, instance.parameter)
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
            if source_type == None or source_type == InstanceType.EDGE_SERVER:
                if msg.get_payload_type() == PayloadType.GRADIENT:
                    # used for both Cloud and Edge
                    with instance.cv:
                        instance.accumulative_gradients.append(msg.payload)
                        instance.cv.notify()
                elif msg.get_payload_type() == PayloadType.REQUEST:
                    # used for Edge Server (when workers ask for parameters)
                    send_message(conn, instance.type,
                                 PayloadType.PARAMETER, instance.parameter)
        if not persistent_connection:
            break


def client_build_connection(host, port, wait_initial_msg=True):
    # create an INET, STREAMing socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    except socket.error:
        print('Failed to create socket')
        sys.exit()

    remote_ip = host

    # Connect to remote server
    try:
        s.connect((remote_ip, port))
    except Exception as e:
        print(f"Connection error: {e}")
        sys.exit(1)

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
