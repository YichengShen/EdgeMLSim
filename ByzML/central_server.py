import tcp
from mxnet import nd



class Central_Server:
    """
    Class for Central Server
    
    Attributes:
        - model
        - gradients_received
    """

    def __init__(self, host='localhost', port=9999):
        """
        The constructor for Central_Server class.

        Parameters:
            - 
        """
        self.model = [nd.random_normal(0,1,shape=(128,784))] +\
                            [nd.random_normal(0,1,shape=(128))] +\
                            [nd.random_normal(0,1,shape=(64,128))] +\
                            [nd.random_normal(0,1,shape=(64))] +\
                            [nd.random_normal(0,1,shape=(10,64))] +\
                            [nd.random_normal(0,1,shape=(10))]
        self.gradients_received = []

    # def send_model(self, connection):
    #     """
    #     Send model to edge servers.

    #     Parameters:
    #         - model:

    #     Returns:

    #     """
    #     pass

    def collect(self, gradients):
        """
        Collect returned gradients from edge servers.

        Parameters:
            - 

        Returns:

        """
        self.gradients_received.append(gradients)
        # print(self.gradients_received)
        # print("Gradients received")
        # print('-----------------------------------------------')
        # print(f"Number of gradients received: {len(self.gradients_received)}")

    def aggregate(self):
        pass

    # def stop(self):
    #     """
    #     Send a stop signal to edge servers.

    #     Parameters:
    #         - 

    #     Returns:

    #     """
    #     pass


if __name__ == "__main__":
    host = 'localhost'
    port = 9999
    central_server = Central_Server()
    tcp.run_server(host, port, central_server)