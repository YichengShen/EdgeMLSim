"""
Before running this file, you need to
   1) Create a Docker swarm on your manager VM ---> `sudo docker swarm init`
   2) Let the worker VMs join the swarm
"""

import docker
import yaml
from time import sleep
from build_image import build_image


def create_overlay_net(client):
    """
    Creates an overlay network with subnet address of 192.168.0.0/24. EdgeMLSim components use IP in this subnet to build TCP connections.
    """
    # Remove previous overlay network if it exists
    try:
        previous_overlay = client.networks.get("overlay_net")
    except docker.errors.NotFound as exc:
        print(f"Exception: {exc.explanation}")
    else:
        previous_overlay.remove()
    # Create overlay network
    ipam_pool = docker.types.IPAMPool(
        subnet='192.168.0.0/24', gateway='192.168.0.1')
    ipam_config = docker.types.IPAMConfig(pool_configs=[ipam_pool])
    overlay_net = client.networks.create(
        "overlay_net", driver="overlay", ipam=ipam_config, attachable=True)
    return overlay_net


def run_simulator(client, image_tag, ip_config, overlay_net):
    """
    Run the Simulator container
    """
    simulator = client.containers.create(
        image_tag, name="simulator", command="python3 Simulator.py", detach=True, tty=True)
    overlay_net.connect(simulator, ipv4_address=ip_config['ip_sim'])
    simulator.start()


def run_cloud_server(client, image_tag, ip_config, overlay_net):
    """
    Run the Cloud Server container
    """
    cloud_server = client.containers.create(
        image_tag, name="cloud", command="python3 CloudServer.py", detach=True, tty=True)
    overlay_net.connect(cloud_server, ipv4_address=ip_config['ip_cloud'])
    cloud_server.start()


def run_edge_servers(client, image_tag, ip_config, overlay_net):
    """
    Run Edge Server containers
    """
    edge_servers = []
    for idx in range(cfg['num_edges']):
        edge_server = client.containers.create(image_tag, name="edge{idx}".format(
            idx=idx), command="python3 EdgeServer.py --ip_index {idx}".format(idx=idx), detach=True, tty=True)
        overlay_net.connect(
            edge_server, ipv4_address=ip_config['ip_edges'][idx])
        edge_server.start()
        edge_servers.append(edge_server)


def run_workers(client, image_tag, overlay_net):
    """
    Run Worker containers
    """
    workers = []
    for idx in range(cfg['num_workers']):
        worker = client.services.create(image_tag,
                                        command="python3 Worker.py",
                                        name="worker{idx}".format(idx=idx),
                                        networks=[overlay_net.id])
        workers.append(worker)


def run_all_components(client, image_tag, ip_config, overlay_net, seconds_sleep=10):
    run_simulator(client, image_tag, ip_config, overlay_net)
    sleep(seconds_sleep)
    run_cloud_server(client, image_tag, ip_config, overlay_net)
    sleep(seconds_sleep)
    run_edge_servers(client, image_tag, ip_config, overlay_net)
    sleep(seconds_sleep)
    run_workers(client, image_tag, overlay_net)
    print("All components running")


if __name__ == "__main__":
    cfg = yaml.load(open('config/config.yml', 'r'), Loader=yaml.FullLoader)

    client = docker.from_env()

    image_tag, ip_config = build_image(client, cfg)

    overlay_net = create_overlay_net(client)

    run_all_components(client, image_tag, ip_config,
                       overlay_net, seconds_sleep=10)
