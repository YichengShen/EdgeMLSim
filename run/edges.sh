#!/bin/bash

source shell_utils/read_config.sh


CONFIG_PATH="config/config.yml"
eval $(parse_yaml $CONFIG_PATH "cfg_")


# Run all edge servers
trap 'kill $(jobs -p)' SIGINT SIGTERM EXIT

declare -i FIRST_PORT=60000
for (( counter=0; counter<$cfg_num_edges; counter++ ))
do
declare -i current_port=$((FIRST_PORT + counter))
# Free up port
fuser -k $current_port/tcp
python3.8 EdgeServer.py --port_index $counter &
if [ $current_port == $FIRST_PORT ]
then
    edge_pid=$!
fi
sleep 0.1
done

wait $edge_pid