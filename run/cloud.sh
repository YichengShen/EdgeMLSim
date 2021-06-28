#!/bin/bash

source shell_utils/read_config.sh


CONFIG_PATH="config/config.yml"
eval $(parse_yaml $CONFIG_PATH "cfg_")

# Free up ports
fuser -k $((cfg_cloud_port))/tcp

# Run Cloud Server
trap 'kill $(jobs -p)' SIGINT SIGTERM EXIT

python3 CloudServer.py &
cloud_pid=$!

wait $cloud_pid