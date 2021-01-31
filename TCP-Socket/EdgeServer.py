import socket
import sys
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
        self.port = None
        self.cv = threading.Condition()
        self.terminated = False
        self.connections = []

    def process(self):
        HOST = socket.gethostname()
        # when PORT is 0, OS picks an available port for you in the bind step
        PORT = 0

        # Build connection with Simulator
        PORT_SIM = SIM_PORT_EDGE
        simulator_conn = client_build_connection(HOST, PORT_SIM, wait_initial_msg=False)
        print('connection with simulator established')

        # Wait for port of Cloud Server sent from Simulator
        port_msg = wait_for_message(simulator_conn)
        cloud_port = port_msg.get_payload()

        # Keep waiting for closing signal from Simulator
        threading.Thread(target=self.wait_to_close, args=(simulator_conn, )).start()

        # build_connection with cloud server
        central_server_conn, msg = client_build_connection(HOST, cloud_port)
        # central_server_conn.settimeout(20)
        self.parameter = msg.get_payload()

        # Keep waiting for new parameters from the central server
        threading.Thread(target=self.receive_parameter, args=(central_server_conn, )).start()
        
        # Start server and wait for workers to connect
        threading.Thread(target=server_handle_connection, args=(HOST, PORT, self, True)).start()
        print("\nEdge Server listening\n")

        # wait for at least num_of_workers workers to join
        # when a worker joins, we send a parameter to the worker
        with self.cv:
            while len(self.connections) < self.cfg['num_workers']:
                self.cv.wait()
        print(f"\n>>> All {len(self.connections)} workers connected \n")

        while True:

            with self.cv:
                while not self.terminated and len(self.accumulative_gradients) < self.cfg['max_edge_gradients']:
                    self.cv.wait()
            # print('received responses from workers')

            if self.terminated:
                break

            # Aggregate
            aggregated_gradient = self.aggregate()

            # Send aggregated gradients to server
            send_message(central_server_conn, InstanceType.EDGE_SERVER, PayloadType.GRADIENT, aggregated_gradient)
            # print('sent aggregated gradients to central server')

        central_server_conn.close()

    def receive_parameter(self, central_server_conn):
        # Used for the thread that waits for parameters sent from the cloud server
        while not self.terminated:
            try: 
                msg = wait_for_message(central_server_conn)
            except OSError:
                sys.exit()
            if not self.terminated:
                self.parameter = msg.get_payload()

    def aggregate(self):
        gradients_to_aggregate = self.accumulative_gradients[:self.cfg['max_edge_gradients']]
        self.accumulative_gradients = self.accumulative_gradients[self.cfg['max_edge_gradients']:]

        # X is a 2d list of nd array
        param_list = [nd.concat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients_to_aggregate]
        mean_nd = nd.mean(nd.concat(*param_list, dim=1), axis=-1)
        grad_collect = []
        idx = 0

        for j, (param) in enumerate(gradients_to_aggregate[0]):
            # mapping back to the collection of ndarray
            # append to list for uploading to cloud
            grad_collect.append(mean_nd[idx:(idx+param.size)].reshape(param.shape))
            idx += param.size
        return grad_collect

    def wait_to_close(self, conn):
        while not self.terminated:
            msg = wait_for_message(conn)
            if msg.get_payload_type() == PayloadType.CONNECTION_SIGNAL:
                self.terminated = True
                conn.close()
                with self.cv:
                    self.cv.notify()


if __name__ == "__main__":
    edge_server = EdgeServer()
    edge_server.process()