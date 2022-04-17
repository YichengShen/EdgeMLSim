import argparse
import sys
import mxnet as mx
import numpy as np
from mxnet import nd, autograd, gluon
import threading
import yaml
from Msg import *
from Utils import *
import CloudServer
from config import config_ml


class EdgeServer:
    def __init__(self, ip_idx):
        # Config
        self.cfg = yaml.load(open('config/config.yml', 'r'),
                             Loader=yaml.FullLoader)
        self.ip_cfg = yaml.load(
            open('deployment/ip_config.yml', 'r'), Loader=yaml.FullLoader)

        # ML Attributes
        self.parameter = None
        self.gradient = 0
        self.accumulative_gradients = []

        # TCP Attributes
        self.type = InstanceType.EDGE_SERVER
        self.ip = self.ip_cfg["ip_edges"][ip_idx]
        self.port = self.ip_cfg["port_edge"]
        self.cv = threading.Condition()
        self.terminated = False
        self.connections = []

    def process(self):

        # Build connection with Simulator
        simulator_conn = client_build_connection(
            self.ip_cfg["ip_sim"], self.ip_cfg["port_sim_edge"], wait_initial_msg=False)
        print('connection with simulator established')

        # Keep waiting for closing signal from Simulator
        threading.Thread(target=self.wait_to_close,
                         args=(simulator_conn, )).start()

        # build_connection with cloud server
        central_server_conn, msg = client_build_connection(
            self.ip_cfg["ip_cloud"], self.ip_cfg["port_cloud"])
        self.parameter = msg.get_payload()

        # Keep waiting for new parameters from the central server
        threading.Thread(target=self.receive_parameter,
                         args=(central_server_conn, )).start()

        # Start server and wait for workers to connect
        threading.Thread(target=server_handle_connection, args=(
            self.ip, self.port, self, True, self.type)).start()
        print("\nEdge Server listening\n")

        # wait for at least num_of_workers workers to join
        # when a worker joins, we send a parameter to the worker
        with self.cv:
            self.cv.wait_for(lambda: len(self.connections)
                             >= self.cfg['num_workers'])
            print(f"\n>>> All {len(self.connections)} workers connected \n")

        while True:

            with self.cv:
                self.cv.wait_for(lambda: self.terminated or config_ml.edge_aggregation_condition(
                    self.accumulative_gradients))
            # print('received responses from workers')

            if self.terminated:
                break

            # Aggregate
            aggregated_gradient = self.aggregate()

            # Send aggregated gradients to server
            send_message(central_server_conn, InstanceType.EDGE_SERVER,
                         PayloadType.GRADIENT, aggregated_gradient)
            # print('sent aggregated gradients to central server')

        central_server_conn.close()

    def receive_parameter(self, central_server_conn):
        # Used for the thread that waits for parameters sent from the cloud server
        while not self.terminated:
            try:
                msg = wait_for_message(central_server_conn)
            except OSError:
                sys.exit()
            if not self.terminated:
                self.parameter = msg.get_payload()

    def aggregate(self):
        gradients_to_aggregate = self.accumulative_gradients[:self.cfg['max_edge_gradients']]
        self.accumulative_gradients = self.accumulative_gradients[self.cfg['max_edge_gradients']:]

        aggregated_nd = config_ml.aggre(
            gradients_to_aggregate, byz=config_ml.BYZ_TYPE_EDGE)

        grad_collect = []
        idx = 0
        for param in gradients_to_aggregate[0]:
            # mapping back to the collection of ndarray
            # append to list for uploading to cloud
            grad_collect.append(
                aggregated_nd[idx:(idx+param.size)].reshape(param.shape))
            idx += param.size
        return grad_collect

    def wait_to_close(self, conn):
        while not self.terminated:
            msg = wait_for_message(conn)
            if msg.get_payload_type() == PayloadType.CONNECTION_SIGNAL:
                self.terminated = True
                conn.close()
                with self.cv:
                    self.cv.notify()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ip_index", help="IP index in the generated IP config", type=int)
    args = parser.parse_args()

    edge_server = EdgeServer(args.ip_index)
    edge_server.process()
