import socket
import threading
import yaml
import random
import math
import mxnet as mx
import numpy as np
from mxnet import nd, autograd, gluon
from mxnet.gluon.data.vision import transforms
from gluoncv.data import transforms as gcv_transforms

from Msg import *
from Worker import Worker
from Utils import *
from locationPicker_v3 import output_junctions
import xml.etree.ElementTree as ET


class Simulator:
    """
    The Simulator that keeps track of traffic info and controls Cloud Server, Edge Server, and Workers.
    1. Run TCP servers and wait for cloud, edge servers, and workers to connect
    2. Start assigning training tasks
        a. In each epoch, shuffle data
            - For each data batch in the epoch, send an edge server port number and the data to a worker
    """
    def __init__(self):
        # Config
        self.cfg = yaml.load(open('config.yml', 'r'), Loader=yaml.FullLoader)

        # ML attributes
        self.epoch = 0
        self.train_data = None
        self.val_train_data = None
        self.val_test_data = None
        self.shuffled_data = []
        self.load_data()
        self.epoch_loss = mx.metric.CrossEntropy()
        self.epoch_accuracy = mx.metric.Accuracy()

        # TCP attributes
        self.type = InstanceType.SIMULATOR
        self.cv = threading.Condition()
        self.cv_main = threading.Condition()
        self.terminated = False
        self.cloud_conn = None
        self.edge_conns = []
        self.worker_count = 0
        self.worker_conns = []
        self.worker_id_free = []

        # Simulation (traffic) attributes
        self.vehicle_dict = {}
        self.edge_locations = {self.cfg['edge_ports'][i]: coordinates for i, coordinates in enumerate(output_junctions)}

    def transform(self, data, label):
        data = data.astype(np.float32) / 255
        return data, label

    def load_data(self):
        """
        Users can change dataset here.
        """
        batch_size = self.cfg['batch_size']
        num_training_data = self.cfg['num_training_data']
        
        # Load Data
        self.train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST('../data/mnist', train=True, transform=self.transform).take(num_training_data),
                                batch_size, shuffle=True, last_batch='discard')
        self.val_train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST('../data/mnist', train=True, transform=self.transform).take(self.cfg['num_val_loss']),
                                    batch_size, shuffle=False, last_batch='keep')
        self.val_test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST('../data/mnist', train=False, transform=self.transform),
                                    batch_size, shuffle=False, last_batch='keep')

    def new_epoch(self):
        self.epoch += 1
        
        # Shuffle data before each new epoch
        for i, (data, label) in enumerate(self.train_data):
            self.shuffled_data.append((data, label))

    def get_model(self):
        send_message(self.cloud_conn, InstanceType.SIMULATOR, PayloadType.REQUEST, b'ask for model')
        model_msg = wait_for_message(self.cloud_conn)
        return model_msg.get_payload()

    def get_accu_loss(self):
        model = self.get_model()
        # Calculate accuracy on testing data
        for i, (data, label) in enumerate(self.val_test_data):
            outputs = model(data)
            self.epoch_accuracy.update(label, outputs)
        # Calculate loss (cross entropy) on training data
        for i, (data, label) in enumerate(self.val_train_data):
            outputs = model(data)
            self.epoch_loss.update(label, nd.softmax(outputs))


    def print_accu_loss(self):
        self.epoch_accuracy.reset()
        self.epoch_loss.reset()
        print("finding accu and loss ...")

        # Calculate accuracy and loss
        self.get_accu_loss()

        _, accu = self.epoch_accuracy.get()
        _, loss = self.epoch_loss.get()

        print("Epoch {:03d}: Loss: {:03f}, Accuracy: {:03f}\n".format(self.epoch,
                                                                            loss,
                                                                            accu))

    def wait_for_free_worker_id(self, worker_conn, id):
        # while not self.terminated:
        id_msg = wait_for_message(worker_conn)
        self.worker_id_free.append(id_msg.get_payload())
        self.vehicle_dict[id] = False
        with self.cv_main:
            self.cv_main.notify()  

    def get_closest_edge_server_port(self, vehicle_x, vehicle_y):
        shortest_distance = 99999999 # placeholder (a random large number)
        closest_edge_server_port = None
        for port, (x, y) in self.edge_locations.items():
            distance = math.sqrt((x - vehicle_x) ** 2 + (y - vehicle_y) ** 2)
            if distance <= self.cfg['v2rsu'] and distance < shortest_distance:
                shortest_distance = distance
                closest_edge_server_port = port
        return closest_edge_server_port

    def process(self):
        """
            loop through sumo file
        """

        HOST = socket.gethostname()

        # Simulator listens for Cloud
        cloud_conn_thread = threading.Thread(target=server_handle_connection, 
                                             args=(HOST, self.cfg["sim_port_cloud"], self, True, self.type, InstanceType.CLOUD_SERVER))
        cloud_conn_thread.start()

        # Simulator listens for Edge Servers
        cloud_conn_thread = threading.Thread(target=server_handle_connection, 
                                             args=(HOST, self.cfg["sim_port_edge"], self, True, self.type, InstanceType.EDGE_SERVER))
        cloud_conn_thread.start()

        # Simulator starts to listen for Workers
        connection_thread = threading.Thread(target=server_handle_connection, 
                                             args=(HOST, self.cfg["sim_port_worker"], self, True, self.type, InstanceType.WORKER))
        connection_thread.start()

        print("\nSimulator listening\n")

        # Wait for cloud to connect
        with self.cv:
            while self.cloud_conn is None:
                self.cv.wait()
        print(f"\n>>> Cloud Server connected \n")

        # Wait for edge servers to connect
        with self.cv:
            while len(self.edge_conns) < self.cfg['num_edges']:
                self.cv.wait()
        print(f"\n>>> All {len(self.edge_conns)} edge servers connected \n")

        # Wait for all workers to connect
        with self.cv:
            while len(self.worker_conns) < self.cfg['num_workers']:
                self.cv.wait()
        print(f"\n>>> All {len(self.worker_conns)} workers connected \n")

        # Parse map xml file
        tree = ET.parse(self.cfg["FCD_FILE"])
        root = tree.getroot()

        self.new_epoch()
        # Maximum training epochs
        while self.epoch <= self.cfg['num_epochs']:
            # For each time step in the FCD file
            for timestep in root[1000:]:
                if self.epoch > self.cfg['num_epochs']:
                    break
            
                # For each vehicle on the map at the timestep
                for vehicle in timestep.findall('vehicle'):
                    # If vehicle not yet stored in vehicle_dict
                    if vehicle.attrib['id'] not in self.vehicle_dict:
                        self.vehicle_dict[vehicle.attrib['id']] = False
                    # If True, the vehicle is already assigned a worker
                    elif self.vehicle_dict[vehicle.attrib['id']]:
                        continue

                    edge_port = self.get_closest_edge_server_port(float(vehicle.attrib['x']), float(vehicle.attrib['y']))
                    # Vehicle not in range of any edge server
                    if edge_port is None:
                        continue
                    else:
                        data = self.shuffled_data.pop()
                        worker_conn = self.worker_conns[self.worker_id_free.pop()]
                        send_message(worker_conn, InstanceType.SIMULATOR, PayloadType.DATA, (edge_port, data))
                        self.vehicle_dict[vehicle.attrib['id']] = True

                        # Wait for the work to finish and send back its id in a new thread
                        threading.Thread(target=self.wait_for_free_worker_id, args=(worker_conn, vehicle.attrib['id'] )).start()

                    with self.cv_main:
                        while len(self.worker_id_free) == 0:
                            self.cv_main.wait()
                 
                    # Run out of training data for the particular epoch
                    if not self.shuffled_data:
                        if self.epoch > 0:
                            self.print_accu_loss()
                        self.new_epoch()
                        if self.epoch > self.cfg['num_epochs']:
                            break            

        # Close the connections with workers
        for worker_conn in self.worker_conns:
            send_message(worker_conn, InstanceType.SIMULATOR, PayloadType.CONNECTION_SIGNAL, b'1')
            worker_conn.close()

        # Close the connections with edge servers
        for edge_conn in self.edge_conns:
            send_message(edge_conn, InstanceType.SIMULATOR, PayloadType.CONNECTION_SIGNAL, b'1')
            worker_conn.close()

        # Close the connections with cloud server
        send_message(self.cloud_conn, InstanceType.SIMULATOR, PayloadType.CONNECTION_SIGNAL, b'1')
        self.cloud_conn.close()

        self.connections = []

        self.terminated = True


if __name__ == "__main__":
    simulator = Simulator()
    simulator.process()