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
    - Change ip address (`sim_ip`, `cloud_ip`, `edge_ip`) to match the ip of your VMs.
    - Change `round` to record results.
    
3. Run (Similar to running Python files seperately on local machines)
    ```
    tmux
    python3.8 Simulator.py
    python3.8 CloudServer.py
    ...
    ```

