import socket
import mxnet as mx
import numpy as np
from mxnet import nd, autograd, gluon
import threading
import yaml
from Msg import *
from Utils import *
import CloudServer


class EdgeServer:
    def __init__(self):
        # Config
        self.cfg = yaml.load(open('config.yml', 'r'), Loader=yaml.FullLoader)

        # ML Attributes
        self.parameter = None
        self.gradient = 0
        self.accumulative_gradients = []

        # TCP Attributes
        self.type = InstanceType.EDGE_SERVER
        self.cv = threading.Condition()
        self.terminated = False
        self.connections = []

    def process(self):
        HOST = socket.gethostname()
        PORT = EDGE_PORT
        CLOUD_SERVER_PORT = SERVER_PORT

        # build_connection with cloud server
        central_server_conn, msg = client_build_connection(HOST, CLOUD_SERVER_PORT)
        self.parameter = msg.get_payload()

        # Keep waiting for new parameters from the central server
        threading.Thread(target=self.receive_parameter, args=((central_server_conn, ))).start()
        
        # Start server and wait for workers to connect
        threading.Thread(target=server_handle_connection, args=(HOST, PORT, self, True)).start()

        # wait for at least num_of_workers workers to join
        # when a worker joins, we send a parameter to the worker
        with self.cv:
            while len(self.connections) < self.cfg['num_workers']:
                self.cv.wait()
        print(f"\n>>> All {len(self.connections)} workers connected \n")

        # Tell workers to start
        for worker_conn in self.connections:
            send_message(worker_conn, InstanceType.EDGE_SERVER, PayloadType.START_MESSAGE, b'start')

        while True:

            with self.cv:
                while len(self.accumulative_gradients) < self.cfg['max_edge_gradients']:
                    self.cv.wait()
            # print('received responses from workers')

            # Aggregate
            aggregated_gradient = self.aggregate()

            # Send aggregated gradients to server
            send_message(central_server_conn, InstanceType.EDGE_SERVER, PayloadType.GRADIENT, aggregated_gradient)
            # print('sent aggregated gradients to central server')

        central_server_conn.close()
        self.terminated = True

    def receive_parameter(self, central_server_conn):
        # Used for the thread that waits for parameters sent from the cloud server
        while True:
            msg = wait_for_message(central_server_conn)
            self.parameter = msg.get_payload()
            # print('received parameter from central server')

            # Upon receving new params from cloud, send them to workers
            self.send_parameter_to_worker()

    def send_parameter_to_worker(self):
        for worker_conn in self.connections:
            send_message(worker_conn, InstanceType.EDGE_SERVER, PayloadType.PARAMETER, self.parameter)

    def aggregate(self):
        # X is a 2d list of nd array
        param_list = [nd.concat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in self.accumulative_gradients]
        mean_nd = nd.mean(nd.concat(*param_list, dim=1), axis=-1)
        grad_collect = []
        idx = 0

        for j, (param) in enumerate(self.accumulative_gradients[0]):
            # mapping back to the collection of ndarray
            # append to list for uploading to cloud
            grad_collect.append(mean_nd[idx:(idx+param.size)].reshape(param.shape))
            idx += param.size
        self.accumulative_gradients = []
        return grad_collect


if __name__ == "__main__":
    edge_server = EdgeServer()
    edge_server.process()