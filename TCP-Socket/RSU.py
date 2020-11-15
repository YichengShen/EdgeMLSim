import socket	#for sockets
import sys	#for exit
import _thread
import pickle
import time
from Utils import *

class RSU:
    HOST = socket.gethostname()
    PORT = 9999

    def __init__(self):
        self.model = 0
        self.gradient = 0

    def process(self):
        HOST = socket.gethostname()	# Symbolic name meaning all available interfaces
        PORT = int(sys.argv[1])	# Arbitrary non-privileged port

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print('Socket created')

        #Bind socket to local host and port
        try:
            s.bind((HOST, PORT))
        except socket.error as msg:
            print('Bind failed. Error :', msg)
            sys.exit()
        print(PORT)
        print('Socket bind complete')

        #Start listening on socket
        s.listen(10)
        print('Socket now listening')

        #now keep talking with the client
        while True:
            #wait to accept a connection - blocking call
            conn, addr = s.accept()
            print('Connected with ' + addr[0] + ':' + str(addr[1]))
            
            #start new thread takes 1st argument as a function name to be run, second is the tuple of arguments to the function.
            _thread.start_new_thread(self.client_thread, (conn,))

        s.close()

    def client_thread(self, worker_conn):
        data = wait_for_message(worker_conn)
        print('gradient received from worker')
        self.gradient = data

        # RSU Logic Here

        # build connection with cloud server
        host = socket.gethostname()
        port = 9999
        cloud_server_conn = build_connection(host, port)

        send_message(pickle.dumps(self.gradient), cloud_server_conn)
        print('gradient sent to Cloud Server')
        
        data = wait_for_message(cloud_server_conn)
        print('gradient received from Cloud Server')
        self.gradient = data

        send_message(pickle.dumps(self.gradient), worker_conn)
        print('gradient sent to worker')

        cloud_server_conn.close()

if __name__ == "__main__":
    rsu = RSU()
    rsu.process()