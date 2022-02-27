from time import sleep
import docker
from ip_generator import generate_ip_config


def build_image(client, cfg):
    """
    Build a new EdgeMLSim image based on freshly generated IP config.

    Arguments:
        client : Docker client from Docker Python SDK
        cfg : The yaml config return by yaml.load()

    Returns:
        image_tag (str) : Name of the EdgeMLSim image hosted on local registry
        ip_config (dict) : A dictionary containing the IP of all nodes
    """

    # Generate a config file containing IP addresses of nodes
    ip_config = generate_ip_config(cfg['num_edges'])
    print("New IP config generated")

    # Build docker image
    image = client.images.build(path=".", tag="edgemlsim:latest")
    image_obj = image[0]
    print("EdgeMLSim image built")

    # Pull registry image hosted on Docker Hub
    registry_image = client.images.pull("registry:2")
    # Remove previous container named "registry" if it exists
    try:
        registry_container = client.containers.get("registry")
    except docker.errors.NotFound as exc:
        print(f"Exception: {exc.explanation}")
    else:
        client.api.remove_container("registry", force=True)
    # Clean up unused containers
    client.containers.prune()
    # Run a local registry for sharing the image
    registry_list = client.api.create_container(registry_image.id,
                                                name="registry",
                                                detach=True,
                                                ports=[5000],
                                                host_config=client.api.create_host_config(port_bindings={
                                                    5000: 5000
                                                }))
    registry_container = client.containers.get(registry_list["Id"])
    registry_container.start()
    print("Local registry created")

    sleep(2)  # Wait for the container to start

    # Push image to local registry
    image_tag = "localhost:5000/edgemlsim"
    image_obj.tag(image_tag)
    server_outputs = client.images.push(image_tag, stream=True, decode=True)
    # for line in server_outputs:
    #     print(line)
    print("Image pushed to local registry\n")

    return image_tag, ip_config
