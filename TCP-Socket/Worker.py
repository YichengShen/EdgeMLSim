import socket	#for sockets
from mxnet import nd, gluon, autograd, init
from Msg import *
from Utils import *
import UtilsSimulator as SimUtil
from NeuralNetwork import Neural_Network
import numpy as np
import time


class Worker:
    def __init__(self):
        # TCP attributes
        self.worker_id = None
        self.start = False
        self.terminated = False

        # ML attributes
        self.model = gluon.nn.Sequential()
        with self.model.name_scope():
            self.model.add(gluon.nn.Dense(128, in_units=784, activation='relu'))
            self.model.add(gluon.nn.Dense(64, in_units=128, activation='relu'))
            self.model.add(gluon.nn.Dense(10, in_units=64))
        self.parameter = None

    def process(self):
        # Build connection with edge server
        host = socket.gethostname()
        PORT_EDGE = EDGE_PORT
    
        edge_server_conn, msg = client_build_connection(host, PORT_EDGE)
        # print('connection with edge server established')

        # Build connection with simulator
        PORT_SIM = SimUtil.SIMULATOR_PORT
        simulator_conn, id_msg = SimUtil.connect_with_simulator(host, PORT_SIM)
        # print('connection with simulator established')
        self.worker_id = id_msg.get_payload()
        # print('id assigned:', self.worker_id)
        
        self.parameter = msg.get_payload()
        # print('received parameter')

        # Wait for the start message from Edge Server
        msg = wait_for_message(edge_server_conn)
        if msg.get_payload_type() == PayloadType.START_MESSAGE:
            # print('start message received')
            self.start = True

        # Keep waiting for new parameters from the edge server
        threading.Thread(target=self.receive_parameter, args=((edge_server_conn, ))).start()

        while self.start:

            # Wait for data from Simulator
            data_msg = wait_for_message(simulator_conn)
            data = data_msg.get_payload()

            # Close connection if closing message received
            if data_msg.get_payload_type() == PayloadType.CONNECTION_SIGNAL:
                simulator_conn.close()
                # print("Done")
                #TODO: find a way to terminate this thread
                self.terminated = True
                break

            # Build a new model using parameters received from edge servers
            self.build_model()
        
            # Compute
            gradients = self.compute(data)
            # print("computed")
            
            try:
                # Send gradients to edge servers
                send_message(edge_server_conn, InstanceType.WORKER, PayloadType.GRADIENT, gradients)
            except:
                return
            # print('gradient sent to edge server')

            # Send msg to Simulator indicating task finished (now ready for new task)
            send_message(simulator_conn, InstanceType.WORKER, PayloadType.ID, self.worker_id)
            # print("finish message with id sent to simulator:", self.worker_id)

    def build_model(self):
        self.model.initialize(init=init.Constant(0), force_reinit=True)
        for i, layer in enumerate(self.model):
            layer.weight.data()[:] = self.parameter[i*2]
            layer.bias.data()[:] = self.parameter[i*2+1]

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

    def receive_parameter(self, edge_server_conn):
        # Used for the thread that waits for parameters sent from the edge server
        while not self.terminated:
            msg = wait_for_message(edge_server_conn)
            self.parameter = msg.get_payload()
            # print('received parameter from edge server')

if __name__ == "__main__":
    worker = Worker()
    worker.process()