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

4. On the manager node, run the install script.

   ```
   bash deployment/install.sh
   ```

5. On VMs that you intend to deploy Docker worker nodes, run the following steps.

   5.1 Install Docker. See [detailed download instructions](https://docs.docker.com/engine/install/ubuntu/).

   ```
   sudo apt-get remove docker docker-engine docker.io containerd runc

   sudo apt-get update
   yes | sudo apt-get install \
      ca-certificates \
      curl \
      gnupg \
      lsb-release

   curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

   echo \
   "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

   sudo apt-get update
   yes | sudo apt-get install docker-ce docker-ce-cli containerd.io docker-compose-plugin
   ```

   5.2 To run Docker without root privileges. See [here](https://docs.docker.com/engine/install/linux-postinstall/).

   ```
   sudo groupadd docker
   sudo usermod -aG docker $USER
   newgrp docker
   ```

   5.3 Install Docker Python SDK.

   ```
   sudo apt install -y python3-pip
   pip3 install docker
   ```

6. On the manager node, update `config_docker.yml`. Change `ip_registry` to be the IP of the VM where you deployed your manager node.

7. Add manager node VM's IP to `etc/docker/daemon.json` on all VMs.

   ```
   cd ../../etc/docker
   sudo nano daemon.json
   ```

   In `daemon.json`, add the following and replace IP_MANAGER_NODE with the actual IP:

   ```
   {
   "insecure-registries" : ["IP_MANAGER_NODE:5000"]
   }
   ```

   After this change, Docker needs to be restarted. Run the following commands:

   ```
   sudo systemctl daemon-reload
   sudo systemctl restart docker
   ```

8. Create a Docker swarm on the manager VM. Then make other worker VMs join the swarm.

   - On manager VM

     ```
     docker swarm init
     ```

     Notice that, after running the command above, Docker prints out a command. Use that command to let worker nodes join the swarm. The command looks like the following

     ```
     docker swarm join --token [TOKEN] [IP]:2377
     ```

   - On each worker VM, run the command.

9. On manager node, edit `config/config.yml`. You can specify how many edge servers and workers to use. The number of each servers and workers will be needed to automatically generate a config for IP addresses.

10. On manager node, run the following command to start the simulation.

```
   python3 deployment/run.py
```

11. To kill, run `bash deployment/kill.sh`

## Notes

1. [Commonly Used Docker Commands](notes/docker_commands.md)
