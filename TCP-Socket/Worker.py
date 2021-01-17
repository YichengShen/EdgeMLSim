import socket	#for sockets
from mxnet import nd, gluon, autograd, init
from Msg import *
from Utils import *
from NeuralNetwork import Neural_Network
import numpy as np
import time


class Worker:
    def __init__(self):
        self.model = gluon.nn.Sequential()
        with self.model.name_scope():
            self.model.add(gluon.nn.Dense(128, in_units=784, activation='relu'))
            self.model.add(gluon.nn.Dense(64, in_units=128, activation='relu'))
            self.model.add(gluon.nn.Dense(10, in_units=64))

    def process(self, data):
        # Build connection with edge server
        host = socket.gethostname()
        port = 6666
    
        edge_server_conn, msg = client_build_connection(host, port)
        # print('connection established')
        
        parameter = msg.get_payload()
        # print('received parameter')

        # Initialize a new model using parameters received from edge servers
        self.model.initialize(init=init.Constant(0), force_reinit=True)
        for i, layer in enumerate(self.model):
            layer.weight.data()[:] = parameter[i*2]
            layer.bias.data()[:] = parameter[i*2+1]
        
        # Compute
        gradients = self.compute(data)
        
        try:
            # Send gradients to edge servers
            send_message(edge_server_conn, InstanceType.WORKER, PayloadType.GRADIENT, gradients)
        except:
            return
        # print('gradient sent to edge server')
        
        # Wait for a confirmation message from edge server
        msg = wait_for_message(edge_server_conn)
        if msg.get_payload_type() == PayloadType.CONNECTION_SIGNAL:
            # print('confirmation received. Closing.')
            edge_server_conn.close()

    def compute(self, data):
        X, y = data

        loss_object = gluon.loss.SoftmaxCrossEntropyLoss()
        with autograd.record():
            output = self.model(X)
            loss = loss_object(output, y)
        loss.backward()

        grad_collect = []
        for param in self.model.collect_params().values():
            if param.grad_req != 'null':
                grad_collect.append(param.grad().copy())

        return grad_collect

    def get_model(self):
        # Used in simulator to access the latest model
        return self.model

if __name__ == "__main__":
    worker = Worker()
    worker.process([1, 2])