import socket	#for sockets
import sys	#for exit
import _thread
import pickle
import time

class RSU:
    def __init__(self):
        self.model = 0
        self.gradient = 0

    def run_server(self):
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

        #Function for handling connections. This will be used to create threads
        def clientthread(conn):
            conn.setblocking(0)
            timeout = 1
            #infinite loop so that function do not terminate and thread do not end.
            while True:
                #Receiving from client
                # data = int.from_bytes(conn.recv(1024), byteorder='big')
                begin = time.time()
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
                if not data: 
                    break

                # self.gradient += data
                print('gradient : ', repr(self.gradient))

                self.run_client()
                conn.setblocking(1)
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

    def run_client(self):
        #create an INET, STREAMing socket
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        except socket.error:
            print('Failed to create socket')
            sys.exit()
            
        print('Socket Created')

        host = socket.gethostname()
        port = 9999

        try:
            remote_ip = socket.gethostbyname(host)

        except socket.gaierror:
            #could not resolve
            print('Hostname could not be resolved. Exiting')
            sys.exit()

        #Connect to remote server
        s.connect((remote_ip , port))

        print('Socket Connected to ' + host + ' on ip ' + remote_ip)

        #Send some data to remote server
        message = self.gradient
        gradient_ = pickle.dumps(self.gradient)

        try :
            #Set the whole string
            s.sendall(gradient_)
        except socket.error:
            #Send failed
            print('Send failed')
            sys.exit()

        print('Message send successfully')

        s.setblocking(0)
        timeout = 1
        begin = None
        #Now receive data
        data = b""
        while True:
            if data and begin is None:
                begin = time.time()
            if data and time.time() - begin > timeout:
                break
            # if time.time() - begin > timeout * 2:
            #     break
            
            try:
                packet = s.recv(4096)
                if packet:
                    data += packet
            except:
                pass
        if data:
            self.gradient = pickle.loads(data)

        print('gradient received from Central Server : ', self.gradient)

        s.close()

if __name__ == "__main__":
    rsu = RSU()

    rsu.run_server()