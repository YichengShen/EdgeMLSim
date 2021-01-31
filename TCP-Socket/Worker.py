import socket
from mxnet import nd, gluon, autograd, init
from Msg import *
from Utils import *
import numpy as np


class Worker:
    def __init__(self):
        # TCP attributes
        self.worker_id = None
        self.edge_conns = {}
        self.terminated = False

        # ML attributes
        self.model = gluon.nn.Sequential()
        with self.model.name_scope():
            self.model.add(gluon.nn.Dense(128, in_units=784, activation='relu'))
            self.model.add(gluon.nn.Dense(64, in_units=128, activation='relu'))
            self.model.add(gluon.nn.Dense(10, in_units=64))
        self.parameter = None

    def process(self):
        host = socket.gethostname()

        # Build connection with simulator
        PORT_SIM = SIM_PORT_WORKER
        simulator_conn, id_msg = client_build_connection(host, PORT_SIM)
        # print('connection with simulator established')
        self.worker_id = id_msg.get_payload()
        # print('id assigned:', self.worker_id)

        # Wait for a list of ports of Edge Servers sent from Simulator
        port_msg = wait_for_message(simulator_conn)
        edge_ports = port_msg.get_payload()
        
        # Build connection with Edge Servers
        for edge_port in edge_ports:
            edge_server_conn = client_build_connection(host, edge_port, wait_initial_msg=False)
            self.edge_conns[edge_port] = edge_server_conn

        while True:

            # Wait for data from Simulator
            data_msg = wait_for_message(simulator_conn)

            # Close connection if closing message received
            if data_msg.get_payload_type() == PayloadType.CONNECTION_SIGNAL:
                simulator_conn.close()
                print("Done")
                #TODO: find a way to terminate this thread
                self.terminated = True
                break

            edge_port, data = data_msg.get_payload()

            # Send msg to Edge Server to ask for parameters
            edge_conn = self.edge_conns[edge_port]
            send_message(edge_conn, InstanceType.WORKER, PayloadType.REQUEST, b'request for parameter')

            # Wait for response from Edge Server
            parameter_msg = wait_for_message(edge_conn)
            self.parameter = parameter_msg.get_payload()
                    

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


if __name__ == "__main__":
    worker = Worker()
    worker.process()