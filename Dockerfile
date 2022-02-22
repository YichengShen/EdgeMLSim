ARG DEBIAN_FRONTEND=noninteractive  
# Check this
FROM ubuntu:20.04
LABEL Name=edgemlsim Version=0.0.1

RUN apt update && \
    apt install -y sudo && \
    sudo DEBIAN_FRONTEND=noninteractive apt install -y nano \
    git \
    gcc \
    software-properties-common

RUN sudo add-apt-repository -y ppa:deadsnakes/ppa && \
    sudo apt update && \
    sudo apt install -y python3.8 \
    python3-pip

RUN mkdir EdgeMLSim
COPY . ./EdgeMLSim
WORKDIR /EdgeMLSim

RUN python3.8 -m pip install -r requirements.txt