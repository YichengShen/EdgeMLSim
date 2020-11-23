import sys
import selectors
import socket
import pickle

def run_server(host, port, instance):
    mysel = selectors.DefaultSelector()
    keep_running = True

    def read(connection, mask):
        "Callback for read events"
        global keep_running

        client_address = connection.getpeername()
        print('read({})'.format(client_address))
        
        data = receive_data(connection)

        if data:
            if data == b'1':
                # Receive model request
                print('Received model request')
                model = pickle.dumps(instance.model)
                connection.setblocking(True)
                connection.sendall(model)
                connection.setblocking(False)
                print('Model sent')
            else:
                # Receive gradients
                instance.collect(pickle.loads(data))
        else:
            # Interpret empty result as closed connection
            print('  closing connection with client')
            mysel.unregister(connection)
            connection.close()
            # Tell the main loop to stop
            keep_running = False


    def accept(sock, mask):
        "Callback for new connections"
        new_connection, addr = sock.accept()
        print('accept({})'.format(addr))
        new_connection.setblocking(False)
        mysel.register(new_connection, selectors.EVENT_READ, read)


    server = create_socket(host, port)
    server.setblocking(False)
    server.bind((host, port))
    server.listen(5)

    mysel.register(server, selectors.EVENT_READ, accept)

    while keep_running:
        for key, mask in mysel.select(timeout=1):
            callback = key.data
            callback(key.fileobj, mask)

    print('shutting down')
    mysel.close()



def run_client(server_host, server_port, instance, message=None):
    mysel = selectors.DefaultSelector()

    # Connecting is a blocking operation, so call setblocking()
    # after it returns.
    sock = create_socket(server_host, server_port)
    sock.connect((server_host, server_port))
    sock.setblocking(False)

    # Set up the selector to watch for when the socket is ready
    # to send data as well as when there is data to read.
    mysel.register(
        sock,
        selectors.EVENT_READ | selectors.EVENT_WRITE,
    )
    keep_running = True
    sent = False # to keep track of only sending once
    while keep_running:
        for key, mask in mysel.select(timeout=1):
            connection = key.fileobj
            # client_address = connection.getpeername()
            # print('client({})'.format(client_address))

            if mask & selectors.EVENT_READ:
                print('  ready to read')
                data = receive_data(connection)
                instance.receive_model(pickle.loads(data))
                keep_running = False
                
            if mask & selectors.EVENT_WRITE and not(sent):
                if message is not None:
                    # is it ok to block when we send?
                    gradients = pickle.dumps(message)
                    send_data(sock, gradients)
                    keep_running = False
                else:
                    send_data(sock, b'1')
                sent = True

    print('shutting down connection')
    mysel.unregister(connection)
    connection.close()
    mysel.close()

def send_data(sock, message):
    sock.setblocking(True)
    sock.sendall(message)
    sock.setblocking(False)

def receive_data(connection):
    data = b""
    while True:
        try:
            packet = connection.recv(4096)
        except:
            break
        if not packet:
            break
        else:
            data += packet
    return data


def create_socket(host, port):
    server_address = (host, port)
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print('starting up on {} port {}'.format(*server_address))
    except socket.error:
        print('Failed to create socket')
        sys.exit()
    return sock

