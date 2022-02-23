"""
Before running this file, you need to 
   1) Create a Docker swarm on your manager VM ---> `sudo docker swarm init`
   2) Let the worker VMs join the swarm
"""

import docker
import yaml
from ip_generator import generate_ip_config


# Generate a config file containing IP addresses of nodes
cfg = yaml.load(open('config/config.yml', 'r'), Loader=yaml.FullLoader)
ip_config = generate_ip_config(cfg['num_edges'])

client = docker.from_env()

# Build docker image


my_net = client.networks.create("my_net", driver="overlay", attachable=True)
print(my_net.attrs)

sim = client.containers.run("1cdfb6dcaa21", name="sim", network="my_net", detach=True, tty=True)

my_net_info = client.api.inspect_network(my_net.id)
client.api.services()

sim_ip = client.api.inspect_container("sim")["NetworkSettings"]["Networks"][str(my_net.name)]["IPAMConfig"]["IPv4Address"]