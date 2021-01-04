import socket	#for sockets
from mxnet import nd, gluon, autograd
from Msg import *
from Utils import *
import time

class Worker:

    def __init__(self):
        self.model = None

    def process(self, data):

        # Build connection with edge server
        host = socket.gethostname()
        port = 6666
        
        # while True: ^^^

        edge_server_conn, msg = client_build_connection(host, port)
        print('connection established')
        
        self.model = msg.get_payload()
        print('received parameter')

        gradients = self.compute(self.model, data)
        
        try:
            send_message(edge_server_conn, InstanceType.WORKER, PayloadType.GRADIENT, gradients)
        except:
            # break ^^^
            return
        print('gradient sent to edge server')

        # Wait for a confirmation message from edge server
        msg = wait_for_message(edge_server_conn)
        if msg.get_payload_type() == PayloadType.CONNECTION_SIGNAL:
            print('confirmation received. Closing.')
            edge_server_conn.close()

        # time.sleep(2)

    def compute(self, model, data):
        X, y = data
        loss_object = gluon.loss.SoftmaxCrossEntropyLoss()

        with autograd.record():
            output = self.model(X)
            loss = loss_object(output, y)
        loss.backward()

        grad_collect = []
        for param in self.model.collect_params().values():
            if param.grad_req != 'null':
                grad_collect.append(param.grad().copy())
        return grad_collect

if __name__ == "__main__":
    worker = Worker()
    worker.process([1, 2])