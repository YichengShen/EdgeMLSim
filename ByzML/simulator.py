from central_server import Central_Server
from edge_server import Edge_Server
from worker import Worker

def simulate():
    # central_server = Central_Server()

    edge_server = Edge_Server()

    edge_server.upload()
    # aggregate
    # edge_server.request_model()
    
    # worker = Worker()
    # edge_server.request_model()
    # worker.request_model()
    # find gradient
    # worker.upload()
    




if __name__ == "__main__":
    simulate()