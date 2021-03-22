#!/bin/bash

parse_yaml() {
    local prefix=$2
    local s='[[:space:]]*' w='[a-zA-Z0-9_]*' fs=$(echo @|tr @ '\034')
    sed -ne "s|^\($s\)\($w\)$s:$s\"\(.*\)\"$s\$|\1$fs\2$fs\3|p" \
            -e "s|^\($s\)\($w\)$s:$s\(.*\)$s\$|\1$fs\2$fs\3|p"  $1 |
    awk -F$fs '{
        indent = length($1)/2;
        vname[indent] = $2;
        for (i in vname) {if (i > indent) {delete vname[i]}}
        if (length($3) > 0) {
            vn=""; for (i=0; i<indent; i++) {vn=(vn)(vname[i])("_")}
            printf("%s%s%s=\"%s\"\n", "'$prefix'",vn, $2, $3);
        }
    }'
}

eval $(parse_yaml config/config.yml "cfg_")

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
