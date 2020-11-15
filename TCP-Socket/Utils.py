import pickle
import socket
import sys
import time

def build_connection(host, port):
    #create an INET, STREAMing socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    except socket.error:
        print('Failed to create socket')
        sys.exit()
        
    print('Socket Created')

    try:
        remote_ip = socket.gethostbyname(host)

    except socket.gaierror:
        #could not resolve
        print('Hostname could not be resolved. Exiting')
        sys.exit()

    #Connect to remote server
    s.connect((remote_ip, port))
    return s

def send_message(data, conn):
    conn.setblocking(1)
    try :
        #Set the whole string
        conn.sendall(data)
    except socket.error:
        #Send failed
        print('Send failed')
        sys.exit()

def wait_for_message(socket):
    socket.setblocking(0)
    timeout = 2
    begin = None
    #Now receive data
    data = b""
    while True:
        if data and begin is None:
            begin = time.time()
        if data and time.time() - begin > timeout:
            break
        # if time.time() - begin > timeout * 2:
        #     break
        try:
            packet = socket.recv(4096)
            if packet:
                data += packet
            else:
                break
        except:
            pass
    return pickle.loads(data)

    socket.close()