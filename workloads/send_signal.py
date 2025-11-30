import socket 
import time

def send_signal(node, port=10000, cmd='test'):
    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(10)  # 10 second timeout for connection and operations

    # Connect the socket to the port where the server is listening
    server_address = (node, int(port))

    print('connecting to {} port {}'.format(*server_address))
    try:
        sock.connect(server_address)
    except (socket.timeout, ConnectionRefusedError, OSError) as e:
        print(f'Failed to connect to {server_address}: {e}')
        sock.close()
        return

    try:
        # Send data
        message = cmd.encode('utf-8') #b'save 35'  #b'start 35 gpu 6'#b'save 35'
 
        print('sending {!r}'.format(message))
        sock.sendall(message)
        
        # Wait for success response with timeout
        start_time = time.time()
        timeout = 10  # 10 second timeout
        while time.time() - start_time < timeout:
            try:
                data = sock.recv(32)
                if data and 'success' in data.decode('utf-8'):
                    #print('received {!r}'.format(data))
                    break
                elif not data:
                    # Connection closed
                    print('Connection closed by server')
                    break
            except socket.timeout:
                print('Timeout waiting for success signal')
                break
        else:
            print('Timeout waiting for success signal after {} seconds'.format(timeout))
    except Exception as e:
        print(f'Error in send_signal: {e}')
    finally:
        #print('closing socket')
        sock.close()
