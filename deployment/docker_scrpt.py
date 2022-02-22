import docker
client = docker.from_env()

my_net = client.networks.create("my_net", driver="overlay", attachable=True)
print(my_net.attrs)

sim = client.containers.run("1cdfb6dcaa21", name="sim", network="my_net", detach=True, tty=True)

my_net_info = client.api.inspect_network(my_net.id)
client.api.services()

sim_ip = client.api.inspect_container("sim")["NetworkSettings"]["Networks"][str(my_net.name)]["IPAMConfig"]["IPv4Address"]