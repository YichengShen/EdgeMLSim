import socket
import sys
import threading
from mxnet import nd
from Msg import *
from Utils import *

class CloudServer:
    def __init__(self):
        self.type = InstanceType.CLOUD_SERVER
        self.parameter = [nd.random_normal(0,1,shape=(128,784))] +\
                    [nd.random_normal(0,1,shape=(128))] +\
                    [nd.random_normal(0,1,shape=(64,128))] +\
                    [nd.random_normal(0,1,shape=(64))] +\
                    [nd.random_normal(0,1,shape=(10,64))] +\
                    [nd.random_normal(0,1,shape=(10))]
        self.buffer = []
        self.cv = threading.Condition()
        self.terminated = False
        self.connections = []
        self.num_edge_servers = 1

    def process(self):
        HOST = socket.gethostname()
        PORT = 9999
        connection_thread = threading.Thread(target=server_handle_connection, args=(HOST, PORT, self, True))
        connection_thread.start()

        with self.cv:
            while len(self.connections) < 1:
                self.cv.wait()

        # for i in range(self.num_epochs):
        #     for conn in self.connections:
        #         send_message(self.connections[0], InstanceType.CLOUD_SERVER, PayloadType.PARAMETER, self.parameter)
        #     print('sent parameter to edge servers')

        # Keep waiting for new gradients
        while True:

            # wait for response from edge servers
            with self.cv:
                while len(self.buffer) < self.num_edge_servers:
                    self.cv.wait()
            print('received responses from edge servers')

            self.parameter = self.aggregate()
            self.send_parameter() # send new parameters to edge servers after aggregation

        self.terminated = True
        print(self.parameter)

    def aggregate(self):
        msg = self.buffer.pop()
        return msg.payload

    def send_parameter(self):
        send_message(self.connections[0], InstanceType.CLOUD_SERVER, PayloadType.PARAMETER, self.parameter)

if __name__ == "__main__":
    cloud_server = CloudServer()
    cloud_server.process()