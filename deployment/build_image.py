from time import sleep
import docker


def build_image(client):
    # Build docker image
    image = client.images.build(path=".", tag="edgemlsim:latest")
    image_obj = image[0]
    image_id = image_obj.short_id[-10:]
    print("EdgeMLSim image built")

    # Run a local registry for sharing the image
    registry_image = client.images.pull("registry:2")
    try:
        registry_container = client.containers.get("registry")
    except docker.errors.NotFound as exc:
        print(f"Check container name!\n{exc.explanation}")
    else:
        client.api.remove_container("registry", force=True)
    client.containers.prune()
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
    for line in client.images.push(image_tag, stream=True, decode=True):
        print(line)
    print("Image pushed to local registry\n")

    return image_id, image_tag
