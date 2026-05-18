
import foxglove
from foxglove import Channel, Schema
from foxglove.channels import CompressedImageChannel
from foxglove.schemas import CompressedImage

import gen.Packet_pb2

from pathlib import Path

import socket
import datetime
import json
import logging
import struct
import time
from math import cos, sin

import numpy as np

def main() -> None:

    UDP_IP = "0.0.0.0"       # Listen on all network interfaces
    UDP_PORT = 30            # Match this to RECV_PORT in your Arduino sketch
    BUFFER_SIZE = 4096         # Should accommodate the full protobuf-encoded packet

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)


    print(f"Listening on port {UDP_PORT}")
    seen_packet_ids = set()

    while True:
            data, _ = sock.recvfrom(1600)
            xml_str = data.decode(errors='replace')
            print("got data")






if __name__ == "__main__":
    main()
