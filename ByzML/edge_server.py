import tcp
from constants import constants
from mxnet import nd


class Edge_Server:
    """
    Class for Edge Server
    
    Attributes:
        - model
        - gradients_received
    """

    def __init__(self):
        """
        The constructor for Edge_Server class.

        Parameters:
            - 
        """
        self.model = None
        self.gradients_received = []
        self.gradients = [nd.random_normal(0,1,shape=(128,784))] +\
                            [nd.random_normal(0,1,shape=(128))] +\
                            [nd.random_normal(0,1,shape=(64,128))] +\
                            [nd.random_normal(0,1,shape=(64))] +\
                            [nd.random_normal(0,1,shape=(10,64))] +\
                            [nd.random_normal(0,1,shape=(10))]

    def upload(self):
        """
        Upload gradients to the central server.

        Parameters:
            - gradients:

        Returns:

        """
        tcp.run_client(constants['HOST'], constants['PORT'], self, self.gradients)

    def request_model(self):
        tcp.run_client(constants['HOST'], constants['PORT'], self)

    def receive_model(self, model):
        print("receive model")
        self.model = pickle.loads(model)
        print(self.model)

    # def send_model(self, connection):
    #     """
    #     Send model to workers.

    #     Parameters:
    #         - connection: 

    #     Returns:

    #     """
    #     pass

    def collect(self, gradients):
        """
        Collect returned gradients from workers.

        Parameters:
            - 

        Returns:

        """
        self.gradients_received.append(gradients)
        pass

    def aggregate(self):
        pass


if __name__ == "__main__":
    host = 'localhost'
    port = 6666
    edge_server = Edge_Server()
    tcp.run_server(host, port, edge_server)