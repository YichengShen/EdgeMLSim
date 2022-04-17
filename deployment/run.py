"""
Before running this file, you need to
   1) Create a Docker swarm on your manager VM ---> `sudo docker swarm init`
   2) Let the worker VMs join the swarm
"""

import docker
import yaml
from time import sleep
from ip_generator import generate_ip_config
from build_image import build_image


def create_overlay_net(client):
    """
    Creates an overlay network with subnet address of 192.168.0.0/24. EdgeMLSim components use IP in this subnet to build TCP connections.
    """
    # Remove previous overlay network if it exists
    try:
        previous_overlay = client.networks.get("overlay_net")
    except docker.errors.NotFound:
        # Ignore the exception if the network called "overlay_net" is not found
        pass
    else:
        previous_overlay.remove()
    # Create overlay network
    ipam_pool = docker.types.IPAMPool(
        subnet='192.168.0.0/24', gateway='192.168.0.1')
    ipam_config = docker.types.IPAMConfig(pool_configs=[ipam_pool])
    overlay_net = client.networks.create(
        "overlay_net", driver="overlay", ipam=ipam_config, attachable=True)
    return overlay_net


cfg = yaml.load(open('config/config.yml', 'r'), Loader=yaml.FullLoader)

# Generate a config file containing IP addresses of nodes
ip_config = generate_ip_config(cfg['num_edges'])
print("New IP config generated")

client = docker.from_env()

image_tag = build_image(client)

overlay_net = create_overlay_net(client)
print("Overlay network created")

# Run the Simulator container
simulator = client.containers.create(
    image_tag, name="simulator", command="python3 Simulator.py", detach=True, tty=True)
overlay_net.connect(simulator, ipv4_address=ip_config['ip_sim'])
simulator.start()

sleep(10)

# Run the Cloud Server container
cloud_server = client.containers.create(
    image_tag, name="cloud", command="python3 CloudServer.py", detach=True, tty=True)
overlay_net.connect(cloud_server, ipv4_address=ip_config['ip_cloud'])
cloud_server.start()

sleep(10)

# Run Edge Server containers
edge_servers = []
for idx in range(cfg['num_edges']):
    edge_server = client.containers.create(image_tag, name="edge{idx}".format(
        idx=idx), command="python3 EdgeServer.py --ip_index {idx}".format(idx=idx), detach=True, tty=True)
    overlay_net.connect(edge_server, ipv4_address=ip_config['ip_edges'][idx])
    edge_server.start()
    edge_servers.append(edge_server)

sleep(10)

# Run Worker as services
workers = []
for idx in range(cfg['num_workers']):
    worker = client.services.create(image_tag,
                                    command="python3 Worker.py",
                                    name="worker{idx}".format(idx=idx),
                                    networks=[overlay_net.id])
    workers.append(worker)

print("All components running")
