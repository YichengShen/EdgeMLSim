# Kill all Docker containers
containers=$(docker ps -q)
if [ ! -z $containers ]; then
    docker kill $containers;
fi

# Quit the Docker swarm
docker swarm leave --force

# Clean up all unused containers, images, etc.
docker system prune --all --force