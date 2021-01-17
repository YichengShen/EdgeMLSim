import socket	#for sockets
import sys	#for exit
import mxnet as mx
import numpy as np
from mxnet import nd, autograd, gluon
import threading
import yaml
from Msg import *
from Utils import *
from NeuralNetwork import Neural_Network
import CloudServer


class EdgeServer:
    def __init__(self):
        self.cfg = yaml.load(open('config.yml', 'r'), Loader=yaml.FullLoader)
        self.type = InstanceType.EDGE_SERVER
        self.parameter = None
        self.gradient = 0
        self.accumulative_gradients = []
        self.cv = threading.Condition()
        self.terminated = False
        self.connections = []

    def process(self):
        HOST = socket.gethostname()
        WORKER_PORT = 6666
        CLOUD_SERVER_PORT = SERVER_PORT

        # build_connection with cloud server
        central_server_conn, msg = client_build_connection(HOST, CLOUD_SERVER_PORT)
        self.parameter = msg.get_payload()

        # Keep waiting for new parameters from the central server
        threading.Thread(target=self.receive_parameter, args=((central_server_conn, ))).start()
        
        # Start server and wait for workers to connect
        threading.Thread(target=server_handle_connection, args=(HOST, WORKER_PORT, self, False)).start()

        while True:
            # wait for at least num_of_workers workers to join
            # when a worker joins, we send a parameter to the worker
            with self.cv:
                while len(self.connections) < self.cfg['num_workers']:
                    self.cv.wait()
            print('enough worker joined')

            with self.cv:
                while len(self.accumulative_gradients) < self.cfg['max_edge_gradients']:
                    self.cv.wait()
            print('received responses from workers')

            # Aggregate
            aggregated_gradient = self.aggregate()

            # Send aggregated gradients to server
            send_message(central_server_conn, InstanceType.EDGE_SERVER, PayloadType.GRADIENT, aggregated_gradient)
            print('sent aggregated gradients to central server')

            # Edge server has to close the connections with workers every time
            for conn in self.connections:
                send_message(conn, InstanceType.EDGE_SERVER, PayloadType.CONNECTION_SIGNAL, b'1')
                conn.close()

            # will a new worker join the connection array during the for loop above? ^^^
            # this new conn will get cleared out ^^^    

            self.connections = []

        central_server_conn.close()
        self.terminated = True

    def receive_parameter(self, central_server_conn):
        # Used for the thread that waits for parameters sent from the cloud server
        while True:
            msg = wait_for_message(central_server_conn)
            self.parameter = msg.get_payload()
            print('received parameter from central server')

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