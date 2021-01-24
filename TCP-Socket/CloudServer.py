import socket
import threading
import mxnet as mx
from mxnet import nd, gluon
import numpy as np
import tensorflow as tf
from Msg import *
from Utils import *
import yaml

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense
from tensorflow.python.keras.layers import deserialize, serialize
from tensorflow.python.keras.saving import saving_utils

class CloudServer:
    def __init__(self):
        # Config
        self.cfg = yaml.load(open('config.yml', 'r'), Loader=yaml.FullLoader)

        # ML attributes
            # Initialize MXNET model
        self.model = gluon.nn.Sequential()
        with self.model.name_scope():
            self.model.add(gluon.nn.Dense(128, in_units=784, activation='relu'))
            self.model.add(gluon.nn.Dense(64, in_units=128, activation='relu'))
            self.model.add(gluon.nn.Dense(10, in_units=64))
        self.model.initialize(mx.init.Xavier(), force_reinit=True)

            # Retreat parameters from initialized model
        grad_collect = []
        for param in self.model.collect_params().values():
            grad_collect.append(param.data())
        self.parameter = grad_collect

        self.accumulative_gradients = []
        
        # TCP attributes
        self.type = InstanceType.CLOUD_SERVER
        self.cv = threading.Condition()
        self.terminated = False
        self.connections = []

    def process(self):
        HOST = socket.gethostname()
        PORT = SERVER_PORT

        # Build connection with Simulator
        PORT_SIM = SIMULATOR_PORT
        simulator_conn = client_build_connection(HOST, PORT_SIM+5, self.type)
        print('connection with simulator established')

        # Run server
        connection_thread = threading.Thread(target=server_handle_connection, args=(HOST, PORT, self, True))
        connection_thread.start()

        # Keep waiting for model request from Simulator
        threading.Thread(target=self.send_model_to_simulator, args=((simulator_conn, ))).start()

        # Wait for all edge servers to connect
        with self.cv:
            while len(self.connections) < self.cfg['num_edges']:
                self.cv.wait()
        print(f"\n>>> All {len(self.connections)} edge servers connected \n")

        # Keep waiting for new gradients
        while True:
            # wait for response from edge servers
            with self.cv:
                while len(self.accumulative_gradients) < self.cfg['max_cloud_gradients']:
                    self.cv.wait()
            # print('received responses from edge servers')

            self.update_model()
            self.send_parameter() # send new parameters to edge servers after aggregation (not model)

        self.terminated = True

    # Update the model with the aggregated gradients from accumulative gradients
    def update_model(self):
        gradients_to_aggregate = self.accumulative_gradients[:self.cfg['max_cloud_gradients']]
        self.accumulative_gradients = self.accumulative_gradients[self.cfg['max_cloud_gradients']:]

        # Aggregate accumulative gradients
        param_list = [nd.concat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients_to_aggregate]
        mean_nd = nd.mean(nd.concat(*param_list, dim=1), axis=-1)

        # Update Model
        idx = 0
        for j, (param) in enumerate(self.model.collect_params().values()):
            if param.grad_req != 'null':
                # mapping back to the collection of ndarray
                # directly update model
                lr = self.cfg['learning_rate']
                param.set_data(param.data() - lr * mean_nd[idx:(idx+param.data().size)].reshape(param.data().shape))
                idx += param.data().size

        # Retreat parameters from the newly updated model
        grad_collect = []
        for param in self.model.collect_params().values():
            grad_collect.append(param.data())
        self.parameter = grad_collect

    def send_parameter(self):
        send_message(self.connections[0], InstanceType.CLOUD_SERVER, PayloadType.PARAMETER, self.parameter)

    def send_model_to_simulator(self, simulator_conn):
        while True:
            model_request_msg = wait_for_message(simulator_conn)
            if model_request_msg.get_payload_type() == PayloadType.REQUEST:
                send_message(simulator_conn, InstanceType.CLOUD_SERVER, PayloadType.MODEL, self.model)

if __name__ == "__main__":
    cloud_server = CloudServer()
    cloud_server.process()