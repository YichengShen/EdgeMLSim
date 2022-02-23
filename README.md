# EdgeMLSim

## To Run

1. Create a number (>1) of VMs on GCP. Use Ubuntu 20.04 LTS. Add your SSH public key to the VMs.

2. SSH into the VMs from your local machine.

   ```
   ssh -i [PATH_TO_PRIVATE_KEY] [USERNAME]@[EXTERNAL_IP]
   ```

3. On the VM you intend to deploy the Docker manager node, download the repository. Then, checkout your desired branch.

   ```
   git clone https://github.com/YichengShen/EdgeMLSim.git
   cd EdgeMLSim
   git checkout docker_version
   ```

4. On all VMs, install Docker. Run the following commands on every VM. See [detailed download instructions](https://docs.docker.com/engine/install/ubuntu/).

   ```
    sudo apt-get update
    sudo apt-get install \
        ca-certificates \
        curl \
        gnupg \
        lsb-release

    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

    echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
    $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io # This took quite long
   ```

5. To run Docker without root privileges. See [here](https://docs.docker.com/engine/install/linux-postinstall/).

   ```
   sudo groupadd docker
   sudo usermod -aG docker $USER
   newgrp docker
   ```

6. Install Docker Python SDK.

   ```
   sudo apt install -y python3-pip
   pip3 install docker
   ```

7. Create a Docker swarm on the manager VM. Then make other worker VMs join the swarm.

   - On manager VM

   ```
   sudo docker swarm init
   ```

   - On worker VMs

   ```
   sudo docker swarm join --token [TOKEN] [IP]:2377
   ```

8. Specify how many edge servers and workers to use in `config/config.yml`. Then generate a IP config.

   ```
   python3 deployment/ip_generator.py
   ```

9. Build a Docker image from the Dockerfile

```
sudo docker build .
```

11. Create an Overlay network.

```
sudo docker network create --driver=overlay --attachable --subnet 192.168.0.0/24 overlay-net
```

12. Run Simulator container
    ```
    sudo docker run --name simulator --network overlay-net --ip 192.168.0.2 -d -t 48775fa03d6b
    ```
