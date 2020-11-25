#Socket client example in python

import socket	#for sockets
from mxnet import nd
from Utils import *

class Worker:

    def __init__(self):
        self.gradient = None

    def process(self):
        # Vehicle logic here
        gradient = [nd.random_normal(0,1,shape=(128,784))] +\
                    [nd.random_normal(0,1,shape=(128))] +\
                    [nd.random_normal(0,1,shape=(64,128))] +\
                    [nd.random_normal(0,1,shape=(64))] +\
                    [nd.random_normal(0,1,shape=(10,64))] +\
                    [nd.random_normal(0,1,shape=(10))]

        # Build connection with RSU
        host = socket.gethostname()
        port = 6666
        edge_server_conn = build_connection(host, port)

        send_message(pickle.dumps(gradient), edge_server_conn)
        print('gradient sent to edge server')

        data = wait_for_message(edge_server_conn)
        print('gradient received from edge server')
        self.gradient = data
        
        edge_server_conn.close()

if __name__ == "__main__":
    worker = Worker()
    worker.process()