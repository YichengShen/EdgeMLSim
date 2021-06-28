# ByzML

## How to run on multiple machines

1. Download and install

    ```
    git clone https://github.com/YichengShen/ByzML.git
    cd ByzML
    git checkout [branchname]
    . install.sh
    ```

2. Modify config

    - In `config/config.yml`, set `local_run: false`.
    - Change ip address (`sim_ip`, `cloud_ip`, `edge_ip`) to match the internal ip of your VMs.
    - Change `round` to record results.
    
3. Run
    - Stay at the root directory of the project
    - In the Simulator VM, run
        ```
        tmux
        . run/sim.sh [num_round]
        ```
    - In the Cloud VM, run
        ```
        tmux
        . run/cloud.sh
        ``` 
    - Do the same for edge servers and workers.

