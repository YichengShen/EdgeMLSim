python_version=python3.8

function install_python() {
    sudo apt update
    sudo apt install -y gcc
    sudo apt install -y software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt update
    sudo apt install -y ${python_version}
    python3.8 --version
}

function install_pip() {
    curl "https://bootstrap.pypa.io/get-pip.py" -o "get-pip.py"
    sudo apt-get install python3.8-distutils
    python3.8 get-pip.py --user
}

function install_dependencies() {
    python3.8 -m pip install -r requirements.txt
}

install_python
install_pip
install_dependencies