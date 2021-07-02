#!/bin/bash

source shell_utils/read_config.sh


CONFIG_PATH="config/config.yml"

eval $(parse_yaml $CONFIG_PATH "cfg_")

trap 'kill $(jobs -p)' SIGINT SIGTERM EXIT

python3 Simulator.py &
simulator_pid=$!

sleep 10

python3 CloudServer.py &

sleep 5

# If you have more than 10 edge servers,
# add more port numbers to config.yml
for (( counter=$cfg_num_edges-1; counter>=0; counter-- ))
do
python3 EdgeServer.py --port_index $counter &
done

sleep 5

for (( counter=$cfg_num_workers-1; counter>=0; counter-- ))
do
python3 Worker.py &
sleep 1
done

wait $simulator_pid
