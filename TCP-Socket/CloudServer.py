import socket
import sys
import _thread
import pickle
import time

class CloudServer:
    def __init__(self):
        self.gradient = None

    def run_server(self):
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

        #Function for handling connections. This will be used to create threads
        def clientthread(conn):
            #infinite loop so that function do not terminate and thread do not end.
            while True:
                conn.setblocking(0)
                timeout = 1
                begin = time.time()
                #Now receive data
                data = b""
                while True:
                    if data and time.time() - begin > timeout:
                        break
                    if time.time() - begin > timeout * 2:
                        break
                    
                    try:
                        packet = conn.recv(4096)
                        if packet:
                            data += packet
                    except:
                        pass
                
                if data:
                    self.gradient = pickle.loads(data)

                print('gradient received from RSU : ', repr(self.gradient))

                # conn.setblocking(1)
                gradient_ = pickle.dumps(self.gradient)
                try :
                    #Set the whole string
                    conn.sendall(gradient_)
                except socket.error:
                    #Send failed
                    print('Send failed')
                    sys.exit()
            #came out of loop
            conn.close()

        #now keep talking with the client
        while True:
            #wait to accept a connection - blocking call
            conn, addr = s.accept()
            print('Connected with ' + addr[0] + ':' + str(addr[1]))
            
            #start new thread takes 1st argument as a function name to be run, second is the tuple of arguments to the function.
            _thread.start_new_thread(clientthread, (conn,))

        s.close()

if __name__ == "__main__":
    cloud_server = CloudServer()
    cloud_server.run_server()