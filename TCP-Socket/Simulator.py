import yaml
import mxnet as mx
import numpy as np
from mxnet import nd, autograd, gluon
from mxnet.gluon.data.vision import transforms
from gluoncv.data import transforms as gcv_transforms

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

    def transform(self, data, label):
        # if cfg['dataset'] == 'cifar10':
        #     data = mx.nd.transpose(data, (2,0,1))
        data = data.astype(np.float32) / 255
        return data, label

    def load_data(self):
        """
        Users can change dataset here.
        """
        batch_size = self.cfg['batch_size']
        num_training_data = self.cfg['num_training_data']
        self.train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST('../data/mnist', train=True, transform=self.transform).take(num_training_data),
                                batch_size, shuffle=True, last_batch='discard')
        self.val_train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST('../data/mnist', train=True, transform=self.transform).take(self.cfg['num_val_loss']),
                                    batch_size, shuffle=False, last_batch='keep')
        self.val_test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST('../data/mnist', train=False, transform=self.transform),
                                    batch_size, shuffle=False, last_batch='keep')

    def new_epoch(self):
        self.epoch += 1
        for i, (data, label) in enumerate(self.train_data):
            self.shuffled_data.append((data, label))
        print(self.epoch)

    def main_loop(self):
        """
            loop through sumo file
        """
        worker = Worker()
        
        while self.epoch <= self.cfg['num_epochs']:

            if not self.shuffled_data:
                self.new_epoch()
            
            worker.process(self.shuffled_data.pop())


if __name__ == "__main__":
    simulator = Simulator()
    simulator.main_loop()