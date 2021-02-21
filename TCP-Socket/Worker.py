import socket
import threading
from mxnet import nd, gluon, autograd, init
from Msg import *
from Utils import *
import numpy as np
import yaml


class Worker:
    def __init__(self):
        # Config
        self.cfg = yaml.load(open('config.yml', 'r'), Loader=yaml.FullLoader)

        # TCP attributes
        self.worker_id = None
        self.edge_port = None
        self.edge_conns = {}
        self.terminated = False
        self.cv_start = threading.Condition()
        self.cv_send = threading.Condition()
        
        # Map attributes
        self.in_map = True

        # ML attributes
        self.model = gluon.nn.Sequential()
        with self.model.name_scope():
            self.model.add(gluon.nn.Dense(128, in_units=784, activation='relu'))
            self.model.add(gluon.nn.Dense(64, in_units=128, activation='relu'))
            self.model.add(gluon.nn.Dense(10, in_units=64))
        self.parameter = None
        self.data = None

    def process(self):
        host = socket.gethostname()

        # Build connection with simulator
        simulator_conn, id_msg = client_build_connection(host, self.cfg["sim_port_worker"])
        # print('connection with simulator established')
        self.worker_id = id_msg.get_payload()
        # print('id assigned:', self.worker_id)
        
        # Build connection with Edge Servers
        for idx in range(self.cfg["num_edges"]):
            edge_port = self.cfg["edge_ports"][idx]
            edge_server_conn = client_build_connection(host, edge_port, wait_initial_msg=False)
            self.edge_conns[edge_port] = edge_server_conn

        threading.Thread(target=self.receive_simulator_info, args=(simulator_conn, )).start()

        while not self.terminated:

            with self.cv_start:
                while self.edge_port is None and self.in_map:
                    self.cv_start.wait()
            print('notified_start')

            # Send msg to Edge Server to ask for parameters
            if self.edge_port is not None:
                edge_conn = self.edge_conns[self.edge_port]
                send_message(edge_conn, InstanceType.WORKER, PayloadType.REQUEST, b'request for parameter')

                # Wait for response from Edge Server
                parameter_msg = wait_for_message(edge_conn)
                self.parameter = parameter_msg.get_payload()

                # Build a new model using parameters received from edge servers
                self.build_model()
        
            # Compute
            if self.data is not None:
                gradients = self.compute(self.data)
            
            with self.cv_send:
                while self.edge_port is None and self.in_map:
                    self.cv_send.wait()
            print("notified_send")
            
            if self.edge_port is not None:
                try:
                    # Send gradients to edge servers
                    send_message(self.edge_conns[self.edge_port], InstanceType.WORKER, PayloadType.GRADIENT, gradients)
                except:
                    return

            # Send msg to Simulator indicating task finished (now ready for new task)
            send_message(simulator_conn, InstanceType.WORKER, PayloadType.ID, self.worker_id)
            # print("finish message with id sent to simulator:", self.worker_id)
            self.data = None
            self.edge_port = None
            self.in_map = True

    def receive_simulator_info(self, simulator_conn):
        while not self.terminated:
            # Wait for data from Simulator
            data_msg = wait_for_message(simulator_conn)

            # Close connection if closing message received
            if data_msg.get_payload_type() == PayloadType.CONNECTION_SIGNAL:
                simulator_conn.close()
                print("Done")
                #TODO: find a way to terminate this thread
                self.terminated = True
                break

            self.edge_port, self.data, self.in_map = data_msg.get_payload()

            if not self.in_map:
                print('not in map')

            if self.data is not None or not self.in_map:
                with self.cv_start:
                    self.cv_start.notify()

            with self.cv_send:
                self.cv_send.notify()

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