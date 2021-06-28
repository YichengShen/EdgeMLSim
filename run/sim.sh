#!/bin/bash

source shell_utils/read_config.sh


CONFIG_PATH="config/config.yml"
eval $(parse_yaml $CONFIG_PATH "cfg_")

# Free up ports
fuser -k $((cfg_sim_port_cloud))/tcp
fuser -k $((cfg_sim_port_edge))/tcp
fuser -k $((cfg_sim_port_worker))/tcp


# Run Simulator
trap 'kill $(jobs -p)' SIGINT SIGTERM EXIT

python3 Simulator.py &
simulator_pid=$!

wait $simulator_pid