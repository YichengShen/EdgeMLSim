import CloudServer, RSU

def main():
    cloud_server = CloudServer.CloudServer()
    # rsu = RSU.RSU()

    cloud_server.run_server()
    # rsu.run_client()

if __name__ == "__main__":
    main()