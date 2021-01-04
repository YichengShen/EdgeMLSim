import yaml
import socket
import sys
import threading
import mxnet as mx
from mxnet import nd, gluon
from Msg import *
from Utils import *

class CloudServer:
    def __init__(self):
        self.cfg = yaml.load(open('config.yml', 'r'), Loader=yaml.FullLoader)
        self.type = InstanceType.CLOUD_SERVER
        self.model = gluon.nn.Sequential()
        with self.model.name_scope():
            self.model.add(gluon.nn.Dense(128, activation='relu'))
            self.model.add(gluon.nn.Dense(64, activation='relu'))
            self.model.add(gluon.nn.Dense(10))
        self.model.initialize(mx.init.Xavier(), force_reinit=True)
        self.accumulative_gradients = []
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
                while len(self.accumulative_gradients) < self.cfg['max_cloud_gradients']:
                    self.cv.wait()
            print('received responses from edge servers')

            self.update_model()
            self.send_parameter() # send new parameters to edge servers after aggregation

        self.terminated = True
        print(self.parameter)

    def aggregate(self):
        msg = self.accumulative_gradients.pop()
        return msg.payload

    # Update the model with its accumulative gradients
    # Used for batch gradient descent
    def update_model(self):
        print("UPDATE")
        param_list = [nd.concat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in self.accumulative_gradients]
        mean_nd = nd.mean(nd.concat(*param_list, dim=1), axis=-1)
        idx = 0
        for j, (param) in enumerate(self.model.collect_params().values()):
            if param.grad_req != 'null':
                # mapping back to the collection of ndarray
                # directly update model
                lr = self.cfg['learning_rate']
                param.set_data(param.data() - lr * mean_nd[idx:(idx+param.data().size)].reshape(param.data().shape))
                idx += param.data().size
        self.accumulative_gradients = []

    def send_parameter(self):
        send_message(self.connections[0], InstanceType.CLOUD_SERVER, PayloadType.PARAMETER, self.model)

if __name__ == "__main__":
    cloud_server = CloudServer()
    cloud_server.process()