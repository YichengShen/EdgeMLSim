#Socket client example in python

import socket	#for sockets
import sys	#for exit
import mxnet as mx
from mxnet import nd
import pickle
import time
from Utils import *

class Vehicle:

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
        rsu_conn = build_connection(host, port)

        send_message(pickle.dumps(gradient), rsu_conn)
        print('gradient sent to RSU')

        data = wait_for_message(rsu_conn)
        print('gradeint received from RSU')
        self.gradient = data
        
        rsu_conn.close()

if __name__ == "__main__":
    rsu = Vehicle()
    rsu.process()