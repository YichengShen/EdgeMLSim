import socket
import sys
import _thread
import pickle
import time
from Utils import *

class CloudServer:

    def __init__(self):
        self.gradient = None

    def process(self):
        HOST = socket.gethostname()	# Symbolic name meaning all available interfaces
        PORT = 9999	# Arbitrary non-privileged port

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print('Socket created')

        #Bind socket to local host and port
        try:
            s.bind((HOST, PORT))
        except socket.error as msg:
            print('Bind failed. Error :', msg)
            sys.exit()
	
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

    def client_thread(self, conn):
        data = wait_for_message(conn)
        print('gradient received from RSU')
        self.gradient = data

        # Cloud Server Logic Here

        send_message(pickle.dumps(self.gradient), conn)
        print('gradient sent to RSU')


if __name__ == "__main__":
    cloud_server = CloudServer()
    cloud_server.process()