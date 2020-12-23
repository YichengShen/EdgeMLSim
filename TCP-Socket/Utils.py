import pickle
import socket
import sys
import threading
import time
from Msg import *


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
                instance.buffer.append(msg)
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
    return s

def send_message(conn, source_type, payload_type, payload):
    msg = Msg(source_type, payload_type, payload)
    data = pickle.dumps(msg)
    try :
        #Set the whole string
        conn.setblocking(True)
        conn.sendall(data)
    except socket.error:
        #Send failed
        print(socket.error)
        print('Send failed')
        sys.exit()

def wait_for_message(conn):
    data = b""
    while True:
        try:
            conn.setblocking(False)
            packet = conn.recv(4096)
            if packet:
                data += packet
            else:
                break
        except:
            if data: 
                break
            else:
                pass
    return pickle.loads(data) if data else None