import socket

def send_udp_packet():
    UDP_IP = "0.0.0.0"  # Localhost (loopback to your listener)
    UDP_PORT = 7070         # Must match the port in your listener
    MESSAGE = "<data><value>123</value></data>"  # Sample XML payload

    # Create the UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Send the message
    sock.sendto(MESSAGE.encode(), (UDP_IP, UDP_PORT))
    print(f"Sent UDP packet to {UDP_IP}:{UDP_PORT} with message: {MESSAGE}")

if __name__ == "__main__":
    send_udp_packet()
