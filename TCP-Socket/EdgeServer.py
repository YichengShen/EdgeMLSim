import socket	#for sockets
import sys	#for exit
import threading
import yaml
from Msg import *
from Utils import *



class EdgeServer:
    def __init__(self):
        file = open('config.yml', 'r')
        self.cfg = yaml.load(file, Loader=yaml.FullLoader)
        self.type = InstanceType.EDGE_SERVER
        self.parameter = 0
        self.gradient = 0
        self.buffer = []
        self.cv = threading.Condition()
        self.terminated = False
        self.connections = []

    def process(self):
        HOST = socket.gethostname()
        WORKER_PORT = 6666
        CLOUD_SERVER_PORT = 9999

        # build_connection with cloud server
        central_server_conn, msg = client_build_connection(HOST, CLOUD_SERVER_PORT)
        self.parameter = msg.get_payload()

        # Keep waiting for new parameters from the central server
        threading.Thread(target=self.receive_parameter, args=((central_server_conn, ))).start()

        threading.Thread(target=server_handle_connection, args=(HOST, WORKER_PORT, self, False)).start()

        while True:

            # wait for at least num_of_workers workers to join
            # when a worker joins, we send a parameter to the worker
            with self.cv:
                while len(self.connections) < self.cfg['num_workers']:
                    self.cv.wait()
            print('enough worker joined')

            # wait for at least num_gradients from workers
            with self.cv:
                while len(self.buffer) < self.cfg['num_gradients']:
                    self.cv.wait()
            print('received responses from workers')

            reduced_gradient = self.reduce()
            aggregated_gradient = self.aggregate(reduced_gradient)

            # send aggregated result to server
            send_message(central_server_conn, InstanceType.EDGE_SERVER, PayloadType.GRADIENT, aggregated_gradient)
            print('sent aggregated result to central server')

            # Edge server has to close the connections with workers every time
            for conn in self.connections:
                send_message(conn, InstanceType.EDGE_SERVER, PayloadType.CONNECTION_SIGNAL, b'1')
                conn.close()

            self.connections = []
            self.buffer = []

        
        central_server_conn.close()
        self.terminated = True

    def receive_parameter(self, central_server_conn):
        while True:
            msg = wait_for_message(central_server_conn)
            self.parameter = msg.get_payload()
            print('received parameter from central server')

    def reduce(self):
        # TODO: replace this
        with self.cv:
            msg = self.buffer.pop()
            return msg.payload

    def aggregate(self, data):
        # TODO: replace this
        return data

if __name__ == "__main__":
    edge_server = EdgeServer()
    edge_server.process()