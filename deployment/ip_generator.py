import yaml

cfg = yaml.load(open('./config/config.yml', 'r'), Loader=yaml.FullLoader)


def enough_ip_in_subnet(num_ip):
    """
    Validates the total number of IP addresses needed.
    Assumes CIDR is /24. The number of usable IP addresses is 252 (from 3 to 254). Returns True if num_ip is less than 254.
    Note: Docker Overlay network has a size limit of up to /24. See https://docs.docker.com/engine/swarm/networking/

    Arguments:
        num_ip (int): The sum of the number of simulator, cloud server, and edge servers

    Returns:
        Boolean
    """
    return num_ip <= 252


def generate_ip_config(num_edge):
    """
    Writes a yaml file, 'config/ip_config.yml', containing the generated IP addresses for the simulator, cloud server, and edge servers. Returns the information as a dict.
    Note: Assumes there are only one simulator and one cloud server.

    Arguments:
        num_edge (int): The number of edge servers

    Returns:
        ip_config (dict) : A dictionary containing the IP of all nodes
    """

    IP_SUBNET = "192.168.0."

    # Assumes only one simulator and one cloud server
    num_ip = num_edge + 1 + 1

    if not enough_ip_in_subnet(num_ip):
        print("Cannot use more than 252 nodes")
        return None

    # 192.168.0.1 is the Gateway
    # 192.168.0.2 will be taken by Docker (Docker's overlay network will create a container called lb-[NetworkName])
    ip_simulator = IP_SUBNET + "3"
    ip_cloud = IP_SUBNET + "4"
    ip_edges = []
    for i in range(5, num_ip+3):
        ip_edges.append(IP_SUBNET+str(i))

    ip_config = {
        'ip_sim': ip_simulator,
        'ip_cloud': ip_cloud,
        'ip_edges': ip_edges,
        'port_sim_cloud': 10000,
        'port_sim_edge': 10001,
        'port_sim_worker': 10002,
        'port_cloud': 50000,
        'port_edge': 60000
    }

    with open('deployment/ip_config.yml', 'w') as file:
        yaml.dump(ip_config, file)

    return ip_config
