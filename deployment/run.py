"""
Before running this file, you need to 
   1) Create a Docker swarm on your manager VM ---> `sudo docker swarm init`
   2) Let the worker VMs join the swarm
"""

import docker
import yaml
from deployment.ip_generator import generate_ip_config


# Generate a config file containing IP addresses of nodes
cfg = yaml.load(open('config/config.yml', 'r'), Loader=yaml.FullLoader)
ip_config = generate_ip_config(cfg['num_edges'])

client = docker.from_env()

# Build docker image
image = client.images.build(path=".")
image_id = image[0].short_id[-10:]

# Create overlay network
ipam_pool = docker.types.IPAMPool(subnet='192.168.0.0/24')
ipam_config = docker.types.IPAMConfig(pool_configs=[ipam_pool])
overlay_net = client.networks.create("overlay_net", driver="overlay", ipam=ipam_config, attachable=True)

# Run the Simulator container
simulator = client.containers.create(image_id, name="simulator", command="python3 Simulator.py", detach=True, tty=True)
overlay_net.connect(simulator, ipv4_address=ip_config['ip_sim'])
simulator.start()
# simulator.exec_run("python3 Simulator.py", detach=True, tty=True)

# my_net_info = client.api.inspect_network(my_net.id)
# client.api.services()

# sim_ip = client.api.inspect_container("sim")["NetworkSettings"]["Networks"][str(my_net.name)]["IPAMConfig"]["IPv4Address"]