import threading
from mxnet import autograd, init
from Msg import *
from Utils import *
from config import config_ml
import yaml


class Worker:
    def __init__(self):
        # Config
        self.cfg = yaml.load(open('config/config.yml', 'r'),
                             Loader=yaml.FullLoader)
        self.ip_cfg = yaml.load(
            open('deployment/ip_config.yml', 'r'), Loader=yaml.FullLoader)

        # TCP attributes
        self.worker_id = None
        self.edge_ip = None
        self.edge_conns = {}
        self.terminated = False
        self.cv = threading.Condition()

        # Map attributes
        self.in_map = True

        # ML attributes
        self.model = config_ml.MODEL
        self.parameter = None
        self.data = None

    def process(self):

        # Build connection with simulator
        simulator_conn, id_msg = client_build_connection(
            self.ip_cfg['ip_sim'], self.ip_cfg['port_sim_worker'])
        # print('connection with simulator established')
        self.worker_id = id_msg.get_payload()

        # Build connection with Edge Servers
        for idx in range(self.cfg["num_edges"]):
            edge_ip = self.ip_cfg['ip_edges'][idx]
            edge_server_conn = client_build_connection(
                edge_ip, self.ip_cfg['port_edge'], wait_initial_msg=False)
            self.edge_conns[edge_ip] = edge_server_conn

        threading.Thread(target=self.receive_simulator_info,
                         args=(simulator_conn, )).start()

        while not self.terminated:

            # Send msg to Edge Server to ask for parameters
            with self.cv:
                self.cv.wait_for(lambda: not self.in_map or (
                    self.edge_ip is not None and self.data is not None) or self.terminated)
                # print('notified_start')

                if self.terminated:
                    break

                if not self.in_map:
                    self.notify_finish(simulator_conn)
                    continue

                edge_conn = self.edge_conns[self.edge_ip]
                send_message(edge_conn, InstanceType.WORKER,
                             PayloadType.REQUEST, b'request for parameter')

            # Wait for response from Edge Server
            parameter_msg = wait_for_message(edge_conn)
            self.parameter = parameter_msg.get_payload()
            # Compute
            gradients = self.compute(self.data)

            # Send gradients to edge servers
            with self.cv:
                self.cv.wait_for(
                    lambda: self.edge_ip is not None or not self.in_map)
                # print("notified_send")

                if not self.in_map:
                    self.notify_finish(simulator_conn)
                    continue

                send_message(
                    self.edge_conns[self.edge_ip], InstanceType.WORKER, PayloadType.GRADIENT, gradients)

            # Finish
            self.notify_finish(simulator_conn)

    def receive_simulator_info(self, simulator_conn):
        while not self.terminated:
            # Wait for data from Simulator
            data_msg = wait_for_message(simulator_conn)

            # Close connection if closing message received
            if data_msg.get_payload_type() == PayloadType.CONNECTION_SIGNAL:
                simulator_conn.close()
                # print("Done")
                self.terminated = True
                with self.cv:
                    self.cv.notify_all()
                break

            _edge_ip, _data, self.in_map = data_msg.get_payload()
            # print('received', _edge_ip, _data==None, self.data==None)
            # Only change data for the first messag
            if self.data is None:
                self.data = _data

            # if not self.in_map:
            #     print('not in map')

            with self.cv:
                self.edge_ip = _edge_ip
                self.cv.notify_all()

            time.sleep(0.01)

    def build_model(self):
        self.model.initialize(init=init.Constant(0), force_reinit=True)
        for i, layer in enumerate(self.model):
            layer.weight.data()[:] = self.parameter[i*2]
            layer.bias.data()[:] = self.parameter[i*2+1]

    def compute(self, data):
        # Build a new model using parameters received from edge servers
        self.build_model()

        X, y = data

        loss_object = config_ml.LOSS
        with autograd.record():
            output = self.model(X)
            loss = loss_object(output, y)
        loss.backward()

        grad_collect = []
        for param in self.model.collect_params().values():
            if param.grad_req != 'null':
                grad_collect.append(param.grad().copy())

        return grad_collect

    def notify_finish(self, simulator_conn):
        # Send msg to Simulator indicating task finished (now ready for new task)
        self.data = None
        self.edge_ip = None
        self.in_map = True
        send_message(simulator_conn, InstanceType.WORKER,
                     PayloadType.ID, self.worker_id)


if __name__ == "__main__":
    worker = Worker()
    worker.process()
