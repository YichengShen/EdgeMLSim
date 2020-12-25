#Socket client example in python

import socket	#for sockets
from mxnet import nd
from Msg import *
from Utils import *
import time

class Worker:

    def __init__(self):
        self.gradient = None

    def process(self):
        gradient = [nd.random_normal(0,1,shape=(128,784))] +\
                    [nd.random_normal(0,1,shape=(128))] +\
                    [nd.random_normal(0,1,shape=(64,128))] +\
                    [nd.random_normal(0,1,shape=(64))] +\
                    [nd.random_normal(0,1,shape=(10,64))] +\
                    [nd.random_normal(0,1,shape=(10))]

        # Build connection with edge server
        host = socket.gethostname()
        port = 6666
        
        while True:
            edge_server_conn, msg = client_build_connection(host, port)
            print('connection established')
            
            parameter = msg.get_payload()
            print('received parameter')
            
            try:
                send_message(edge_server_conn, InstanceType.WORKER, PayloadType.GRADIENT, parameter)
            except:
                break
            print('gradient sent to edge server')

            # Wait for a confirmation message from edge server
            msg = wait_for_message(edge_server_conn)
            if msg.get_payload_type() == PayloadType.CONNECTION_SIGNAL:
                print('confirmation received. Closing.')
                edge_server_conn.close()

            # time.sleep(2)

    def compute(self, parameter):
        # TODO: replace this
        return parameter

if __name__ == "__main__":
    worker = Worker()
    worker.process()