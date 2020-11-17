import tcp
from constants import constants
from mxnet import nd
import pickle


class Worker:
    """
    Class for Worker
    
    Attributes:
        - model
    """

    def __init__(self):
        """
        The constructor for Worker class.

        Parameters:
            - 
        """
        self.model = None
        self.gradients = [nd.random_normal(0,1,shape=(128,784))] +\
                            [nd.random_normal(0,1,shape=(128))] +\
                            [nd.random_normal(0,1,shape=(64,128))] +\
                            [nd.random_normal(0,1,shape=(64))] +\
                            [nd.random_normal(0,1,shape=(10,64))] +\
                            [nd.random_normal(0,1,shape=(10))]
        

    def upload(self, host='localhost', port=6666):
        """
        Upload gradients to the central server.

        Parameters:
            - host:
            - port

        Returns:

        """
        gradients = pickle.dumps(self.gradients)
        tcp.run_client(host, port, self, gradients)

    def request_model(self, host='localhost', port=6666):
        tcp.run_client(host, port, self)

    def receive_model(self, model):
        print("receive model")
        self.model = pickle.loads(model)
        print(self.model)


if __name__ == "__main__":
    worker = Worker()
    tcp.run_client('localhost', 6666, worker)