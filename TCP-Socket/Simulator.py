import yaml
import mxnet as mx
import numpy as np
from mxnet import nd, autograd, gluon
from mxnet.gluon.data.vision import transforms
from gluoncv.data import transforms as gcv_transforms
import tensorflow as tf

from Worker import Worker

class Simulator:
    def __init__(self):
        self.cfg = yaml.load(open('config.yml', 'r'), Loader=yaml.FullLoader)
        self.epoch = 0
        self.train_data = None
        self.val_train_data = None
        self.val_test_data = None
        self.shuffled_data = []
        self.load_data()
        self.epoch_loss = mx.metric.CrossEntropy()
        self.epoch_accuracy = mx.metric.Accuracy()

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


    def get_accu_loss(self, worker):
        model = worker.get_model()
        # Calculate accuracy on testing data
        for i, (data, label) in enumerate(self.val_test_data):
            outputs = model(data)
            self.epoch_accuracy.update(label, outputs)
        # Calculate loss (cross entropy) on training data
        for i, (data, label) in enumerate(self.val_train_data):
            outputs = model(data)
            self.epoch_loss.update(label, nd.softmax(outputs))


    def print_accu_loss(self, worker):
        self.epoch_accuracy.reset()
        self.epoch_loss.reset()
        print("finding accu and loss ...")

        # Calculate accuracy and loss
        self.get_accu_loss(worker)

        _, accu = self.epoch_accuracy.get()
        _, loss = self.epoch_loss.get()

        print("Epoch {:03d}: Loss: {:03f}, Accuracy: {:03f}\n".format(self.epoch,
                                                                            loss,
                                                                            accu))

    def main_loop(self):
        """
            loop through sumo file
        """
        worker = Worker()
        
        while self.epoch <= self.cfg['num_epochs']:
            
            # Run out of training data for the particular epoch
            if not self.shuffled_data:
                if self.epoch > 0:
                    self.print_accu_loss(worker)
                self.new_epoch()

            worker.process(self.shuffled_data.pop())


if __name__ == "__main__":
    simulator = Simulator()
    simulator.main_loop()