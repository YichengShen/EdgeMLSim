# Kill all Docker containers
c=$(docker ps -q) && [[ $c ]] && docker kill $c

# Quit the Docker swarm
docker swarm leave --force

# Clean up all unused containers, images, etc.
docker system prune --all --force