#Socket client example in python

import socket	#for sockets
import sys	#for exit
import mxnet as mx
from mxnet import nd
import pickle
import time

class Vehicle:
    def __init__(self):
        self.gradient = None

    def run_client(self):
        #create an INET, STREAMing socket
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        except socket.error:
            print('Failed to create socket')
            sys.exit()
            
        print('Socket Created')

        host = socket.gethostname()
        port = 6666

        try:
            remote_ip = socket.gethostbyname(host)

        except socket.gaierror:
            #could not resolve
            print('Hostname could not be resolved. Exiting')
            sys.exit()

        #Connect to remote server
        s.connect((remote_ip , port))

        print('Socket Connected to ' + host + ' on ip ' + remote_ip)

        #Send some data to remote server
        gradient = [nd.random_normal(0,1,shape=(128,784))] +\
                    [nd.random_normal(0,1,shape=(128))] +\
                    [nd.random_normal(0,1,shape=(64,128))] +\
                    [nd.random_normal(0,1,shape=(64))] +\
                    [nd.random_normal(0,1,shape=(10,64))] +\
                    [nd.random_normal(0,1,shape=(10))]
        gradient_ = pickle.dumps(gradient)

        try :
            #Set the whole string
            s.sendall(gradient_)
        except socket.error:
            #Send failed
            print('Send failed')
            sys.exit()

        print('Message send successfully')

        s.setblocking(0)
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
                packet = s.recv(4096)
                if packet:
                    data += packet
                else:
                    break
            except:
                pass
        if data:
            self.gradient = pickle.loads(data)

        print('gradient received from RSU : ', repr(self.gradient))

        s.close()

if __name__ == "__main__":
    rsu = Vehicle()
    rsu.run_client()