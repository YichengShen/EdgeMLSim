from central_server import Central_Server
from edge_server import Edge_Server

def simulate():
    # central_server = Central_Server()
    edge_server = Edge_Server()

    # edge_server.request_model()
    edge_server.upload()


if __name__ == "__main__":
    simulate()