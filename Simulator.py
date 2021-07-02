import os
import socket
import threading
import yaml
import argparse
import csv
import random
import math
import time
import mxnet as mx
import numpy as np
from mxnet import nd, autograd, gluon
from mxnet.gluon.data.vision import transforms
from gluoncv.data import transforms as gcv_transforms
import psutil

from Msg import *
from Utils import *
from locationPicker_v3 import output_junctions
import xml.etree.ElementTree as ET
from config import config_ml


class Simulator:
    """
    The Simulator that keeps track of traffic info and controls Cloud Server, Edge Server, and Workers.
    1. Run TCP servers and wait for cloud, edge servers, and workers to connect
    2. Start assigning training tasks
        a. In each epoch, shuffle data
            - For each data batch in the epoch, send an edge server port number and the data to a worker
    """
    def __init__(self, num_round):
        # Config
        self.cfg = yaml.load(open('config/config.yml', 'r'), Loader=yaml.FullLoader)
        self.num_round = num_round

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
        self.lock = threading.Lock()
        self.terminated = False
        self.cloud_conn = None
        self.edge_conns = []
        self.worker_count = 0
        self.worker_conns = []
        self.worker_id_free = set()

        # Simulation (traffic) attributes
        self.vehicle_dict = {}
        self.edge_locations = {self.cfg['edge_ports'][i]: coordinates for i, coordinates in enumerate(output_junctions)}
        self.sumo_root = None
        self.timestep = None
        # Clock attributes
        self.num_timesteps = 0
        self.total_time = 0
        self.pause_clock = False

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
        self.train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST('./data/mnist', train=True, transform=self.transform).take(num_training_data),
                                batch_size, shuffle=True, last_batch='discard')
        self.val_train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST('./data/mnist', train=True, transform=self.transform).take(self.cfg['num_val_loss']),
                                    batch_size, shuffle=False, last_batch='keep')
        self.val_test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST('./data/mnist', train=False, transform=self.transform),
                                    batch_size, shuffle=False, last_batch='keep')

    def new_epoch(self):
        self.epoch += 1
        
        # Shuffle data before each new epoch
        for data, label in self.train_data:
            self.shuffled_data.append((data, label))

        # Print network stats
        # print(psutil.net_io_counters())

    def get_model(self):
        send_message(self.cloud_conn, InstanceType.SIMULATOR, PayloadType.REQUEST, b'ask for model')
        model_msg = wait_for_message(self.cloud_conn)
        return model_msg.get_payload()

    def get_accu_loss(self, model):
        # Calculate accuracy on testing data
        for data, label in self.val_test_data:
            outputs = model(data)
            self.epoch_accuracy.update(label, outputs)
        # Calculate loss (cross entropy) on training data
        for data, label in self.val_train_data:
            outputs = model(data)
            self.epoch_loss.update(label, nd.softmax(outputs))

    def print_accu_loss(self):
        self.epoch_accuracy.reset()
        self.epoch_loss.reset()
        print("finding accu and loss ...")

        model = self.get_model()

        # Calculate accuracy and loss
        self.get_accu_loss(model)

        _, accu = self.epoch_accuracy.get()
        _, loss = self.epoch_loss.get()

        print("Epoch {:03d}: Loss: {:03f}, Accuracy: {:03f}, Time: {} \n".format(self.epoch,
                                                                                    loss,
                                                                                    accu,
                                                                                    self.total_time))
        self.save(model, self.epoch, accu, loss, self.total_time)
    
    def save(self, model, epoch, accu, loss, time):
        # Save model checkpoints
        if self.cfg['save_model_checkpoints']:
            if not os.path.exists('model_checkpoints'):
                os.makedirs('model_checkpoints')
            checkpoint_file_name = self.cfg['dataset'] + '-' + self.cfg['aggregation_method'] + '-' + self.cfg['byzantine_type_edge'] + '-Epoch' + str(epoch) + '-' + str(self.num_round) + '.params'
            checkpoint_path = os.path.join('model_checkpoints', checkpoint_file_name)
            model.save_parameters(checkpoint_path)

        # Save accu, loss, etc
        if self.cfg['save_results']:
            if not os.path.exists('collected_results'):
                os.makedirs('collected_results')
            dir_name = self.cfg['dataset'] + '-' + self.cfg['aggregation_method'] + '-' + self.cfg['byzantine_type_edge'] + '-' + str(self.num_round) + '.csv'
            p = os.path.join('collected_results', dir_name)
            with open(p, mode='a') as f:
                writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow([epoch, accu, loss, time])

    def wait_for_free_worker_id(self, worker_conn, id):
        # while not self.terminated:
        id_msg = wait_for_message(worker_conn)
        self.worker_id_free.add(id_msg.get_payload())
        self.vehicle_dict[id]['training'] = False

    def get_closest_edge_server_port(self, vehicle_x, vehicle_y):
        shortest_distance = 99999999 # placeholder (a random large number)
        closest_edge_server_port = None
        for port, (x, y) in self.edge_locations.items():
            distance = math.sqrt((x - vehicle_x) ** 2 + (y - vehicle_y) ** 2)
            if distance <= self.cfg['v2rsu'] and distance < shortest_distance:
                shortest_distance = distance
                closest_edge_server_port = port
        return closest_edge_server_port

    # Check if the vehicle is still in the map in the next timestep
    def in_map(self, timestep, v_id):
        current_timestep = float(timestep.attrib['time'])
        next_timestep = self.sumo_root.find('timestep[@time="{:.2f}"]'.format(current_timestep+1))
        if next_timestep == None: # reached the end of fcd file
            next_timestep = self.sumo_root.find('timestep[@time="{:.2f}"]'.format(0))
        id_set = set(map(lambda vehicle: vehicle.attrib['id'], next_timestep.findall('vehicle')))
        return v_id in id_set

    def clock(self):
        while not self.terminated:
            self.timestep = self.sumo_root[self.total_time % self.num_timesteps]
            if not self.pause_clock:
                self.total_time += 1
            time.sleep(1)
            # print(len(self.shuffled_data), self.worker_id_free, threading.active_count())


    def process(self):
        """
            loop through sumo file
        """

        if self.cfg["local_run"]:
            HOST = socket.gethostname()
        else:
            HOST = self.cfg["sim_ip"]

        # Simulator listens for Cloud
        threading.Thread(target=server_handle_connection, 
                        args=(HOST, self.cfg["sim_port_cloud"], self, True, self.type, InstanceType.CLOUD_SERVER)).start()

        # Simulator listens for Edge Servers
        threading.Thread(target=server_handle_connection, 
                        args=(HOST, self.cfg["sim_port_edge"], self, True, self.type, InstanceType.EDGE_SERVER)).start()

        # Simulator starts to listen for Workers
        threading.Thread(target=server_handle_connection, 
                        args=(HOST, self.cfg["sim_port_worker"], self, True, self.type, InstanceType.WORKER)).start()

        print("\nSimulator listening\n")

        with self.cv:
            # Wait for cloud to connect
            self.cv.wait_for(lambda: self.cloud_conn is not None)
            print(f"\n>>> Cloud Server connected \n")

            # Wait for edge servers to connect
            self.cv.wait_for(lambda: len(self.edge_conns) >= self.cfg['num_edges'])
            print(f"\n>>> All {len(self.edge_conns)} edge servers connected \n")

            # Wait for all workers to connect
            self.cv.wait_for(lambda: len(self.worker_conns) >= self.cfg['num_workers'])
            print(f"\n>>> All {len(self.worker_conns)} workers connected \n")

        # Parse map xml file
        tree = ET.parse(self.cfg["FCD_FILE"])
        self.sumo_root = tree.getroot()
        self.num_timesteps = len(self.sumo_root)

        # Clock starts ticking
        threading.Thread(target=self.clock).start()

        print("start clock")

        self.new_epoch()

        # Maximum training epochs
        while self.epoch <= self.cfg['num_epochs']: 
            timestep = self.timestep
            vehicle_list = timestep.findall('vehicle')

            # For each vehicle on the map at the timestep
            for vehicle in vehicle_list:
                # If vehicle not yet stored in vehicle_dict
                v_id = vehicle.attrib['id']
                if v_id not in self.vehicle_dict:
                    self.vehicle_dict[v_id] = {'training': False, 'connection': None, 'last_port': None}
                    
                edge_port = self.get_closest_edge_server_port(float(vehicle.attrib['x']), float(vehicle.attrib['y']))
                data = None

                
                # Vehicle does not have training task currently
                if not self.vehicle_dict[v_id]['training']:
                    self.vehicle_dict[v_id]['connection'] = None
                    self.vehicle_dict[v_id]['last_port'] = None

                    with self.lock:
                        # If no free worker, continue
                        if not self.worker_id_free or edge_port is None:
                            continue                  
                        # If free worker available
                        workerId = self.worker_id_free.pop()
                        self.vehicle_dict[v_id]['connection'] = self.worker_conns[workerId]
                        self.vehicle_dict[v_id]['training'] = True
                        # Run out of training data for the particular epoch
                        if not self.shuffled_data:
                            self.pause_clock = True
                            print('------------------start pause-----------------------')
                            if self.epoch > 0:
                                # if self.epoch <= 10 or self.epoch % 10 == 0:
                                self.print_accu_loss()
                            self.new_epoch()
                            self.pause_clock = False
                            print('--------------------end pause-----------------------')
                            if self.epoch > self.cfg['num_epochs']:
                                break
                        data = self.shuffled_data.pop()

                    # Wait for the work to finish and send back its id in a new thread
                    threading.Thread(target=self.wait_for_free_worker_id, args=(self.vehicle_dict[v_id]['connection'], v_id)).start()
                
                worker_conn = self.vehicle_dict[v_id]['connection']

                # in_map returns False when the vehicle is no longer in the map in the next timestep
                in_map = self.in_map(timestep, v_id)
                
                # If vehicle has same port as last time and still in map, continue
                if edge_port == self.vehicle_dict[v_id]['last_port'] and in_map:
                    continue

                self.vehicle_dict[v_id]['last_port'] = edge_port

                # Cases to send msg:
                # 1. First time assigning task
                # 2. Vehicle changes its port (this means leaving edge range or moving to a new edge server)
                # 3. Vehilce leaves map
                if self.vehicle_dict[v_id]['training']:
                    send_message(worker_conn, InstanceType.SIMULATOR, PayloadType.DATA, (edge_port, data, in_map))

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


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model for image classification.')
    parser.add_argument('--num-round', type=int, default=0,
                        help='number of round.')
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_args()
    num_round = opt.num_round

    simulator = Simulator(num_round)
    simulator.process()