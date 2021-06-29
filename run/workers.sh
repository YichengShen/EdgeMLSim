#!/bin/bash

source shell_utils/read_config.sh


CONFIG_PATH="config/config.yml"
eval $(parse_yaml $CONFIG_PATH "cfg_")

# Run all workers
trap 'kill $(jobs -p)' SIGINT SIGTERM EXIT

# $cfg_num_workers
NUM_WORKERS=50
for (( counter=$NUM_WORKERS-1; counter>=0; counter-- ))
do
$cfg_python_version Worker.py &
if [ $counter == $NUM_WORKERS-1 ]
then
    worker_pid=$!
fi
sleep 0.1
done

wait $worker_pid
