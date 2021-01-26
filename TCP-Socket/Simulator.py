import socket
import threading
import yaml
import mxnet as mx
import numpy as np
from mxnet import nd, autograd, gluon
from mxnet.gluon.data.vision import transforms
from gluoncv.data import transforms as gcv_transforms
import tensorflow as tf

from Msg import *
from Worker import Worker
from Utils import *


class Simulator:
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
        self.worker_count = 0
        self.worker_conns = []
        self.worker_id_free = []

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

    def wait_for_free_worker_id(self, worker_conn):
        # while not self.terminated:
        id_msg = wait_for_message(worker_conn)
        self.worker_id_free.append(id_msg.get_payload())
        with self.cv_main:
            self.cv_main.notify()  

    def process(self):
        """
            loop through sumo file
        """

        # Simulator starts to listen for Workers
        HOST = socket.gethostname()
        PORT = SIMULATOR_PORT
        connection_thread = threading.Thread(target=server_handle_connection, 
                                             args=(HOST, PORT, self, True, self.type, InstanceType.WORKER))
        connection_thread.start()

        # Simulator listens for Cloud
        cloud_conn_thread = threading.Thread(target=server_handle_connection, 
                                             args=(HOST, PORT+5, self, True, self.type, InstanceType.CLOUD_SERVER))
        cloud_conn_thread.start()

        #TODO: keep track of edge servers' port/ID 
        # use this to determine which worker should receive data

        # Wait for cloud to connect
        with self.cv:
            while self.cloud_conn is None:
                self.cv.wait()
        print(f"\n>>> Cloud Server connected \n")

        # Wait for all workers to connect
        total_num_workers = self.cfg['num_edges'] * self.cfg['num_workers']
        with self.cv:
            while len(self.worker_conns) < total_num_workers:
                self.cv.wait()
        print(f"\n>>> All {len(self.worker_conns)} workers connected \n")

        # for worker_conn in self.worker_conns:
        #     # Wait for the work to finish and send back its id in a new thread
        #     threading.Thread(target=self.wait_for_free_worker_id, args=(worker_conn, )).start()

        self.new_epoch()
        while self.epoch <= self.cfg['num_epochs']:

            # Run out of training data for the particular epoch
            if not self.shuffled_data:
                if self.epoch > 0:
                    self.print_accu_loss()
                self.new_epoch()
                if self.epoch > self.cfg['num_epochs']:
                    break
            
            data = self.shuffled_data.pop()
            
            with self.cv_main:
                while len(self.worker_id_free) == 0:
                    self.cv_main.wait()
            
            worker_conn = self.worker_conns[self.worker_id_free.pop()]
            send_message(worker_conn, InstanceType.SIMULATOR, PayloadType.DATA, data)

            # Wait for the work to finish and send back its id in a new thread
            threading.Thread(target=self.wait_for_free_worker_id, args=(worker_conn, )).start()

        # Close the connections with workers
        for worker_conn in self.worker_conns:
            send_message(worker_conn, InstanceType.SIMULATOR, PayloadType.CONNECTION_SIGNAL, b'1')
            worker_conn.close()

        self.connections = []

        self.terminated = True


if __name__ == "__main__":
    simulator = Simulator()
    simulator.process()