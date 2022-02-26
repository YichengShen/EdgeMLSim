import docker


def build_image(client):
    # Build docker image
    image = client.images.build(path=".", tag="edgemlsim:latest")
    image_obj = image[0]
    image_id = image_obj.short_id[-10:]
    print("EdgeMLSim image built")

    # Run a local registry for sharing the image
    client.api.create_container("registry:2",
                                name="registry",
                                detach=True,
                                ports=[5000],
                                restart_policy={"Name": "always"},
                                host_config=client.api.create_host_config(port_bindings={
                                    5000: 5000,
                                }))
    print("Local registry created")

    # Push image to local registry
    image_obj.tag("localhost:5000/edgemlsim")
    for line in client.images.push("localhost:5000/edgemlsim", stream=True, decode=True):
        print(line)
    print("Image pushed to local registry\n")

    return image_id
