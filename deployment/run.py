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


cfg = yaml.load(open('config/config.yml', 'r'), Loader=yaml.FullLoader)

# Generate a config file containing IP addresses of nodes
ip_config = generate_ip_config(cfg['num_edges'])
print("New IP config generated")

client = docker.from_env()

image_id = build_image(client)

# # Create overlay network
# ipam_pool = docker.types.IPAMPool(
#     subnet='192.168.0.0/24', gateway='192.168.0.1')
# ipam_config = docker.types.IPAMConfig(pool_configs=[ipam_pool])
# overlay_net = client.networks.create(
#     "overlay_net", driver="overlay", ipam=ipam_config, attachable=True)

# # Run the Simulator container
# simulator = client.containers.create(
#     image_id, name="simulator", command="python3 Simulator.py", detach=True, tty=True)
# overlay_net.connect(simulator, ipv4_address=ip_config['ip_sim'])
# simulator.start()

# sleep(10)

# # Run the Cloud Server container
# cloud_server = client.containers.create(
#     image_id, name="cloud", command="python3 CloudServer.py", detach=True, tty=True)
# overlay_net.connect(cloud_server, ipv4_address=ip_config['ip_cloud'])
# cloud_server.start()

# sleep(10)

# # Run Edge Server containers
# edge_servers = []
# for idx in range(cfg['num_edges']):
#     edge_server = client.containers.create(image_id, name="edge{idx}".format(
#         idx=idx), command="python3 EdgeServer.py --ip_index {idx}".format(idx=idx), detach=True, tty=True)
#     overlay_net.connect(edge_server, ipv4_address=ip_config['ip_edges'][idx])
#     edge_server.start()
#     edge_servers.append(edge_server)

# sleep(10)

# # Run Worker as services
# workers = []
# for idx in range(cfg['num_workers']):
#     client.services.create()

# my_net_info = client.api.inspect_network(my_net.id)
# client.api.services()

# sim_ip = client.api.inspect_container("sim")["NetworkSettings"]["Networks"][str(my_net.name)]["IPAMConfig"]["IPv4Address"]