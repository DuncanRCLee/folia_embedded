
import foxglove
from foxglove import Channel, Schema
from foxglove.channels import CompressedImageChannel
from foxglove.schemas import CompressedImage

import gen.Packet_pb2

from pathlib import Path

import csv
import socket
import datetime
import json
import logging
import struct
import time
from math import cos, sin
import argparse
import sys
import threading
import select

import foxglove
import numpy as np
from foxglove import Channel, Schema
from foxglove.channels import RawImageChannel
from foxglove.schemas import (
    Color,
    CubePrimitive,
    Duration,
    FrameTransform,
    FrameTransforms,
    PackedElementField,
    PackedElementFieldNumericType,
    PointCloud,
    Pose,
    Quaternion,
    RawImage,
    SceneEntity,
    SceneUpdate,
    Timestamp,
    Vector3,
)
from foxglove.websocket import (
    Capability,
    ChannelView,
    Client,
    ClientChannel,
    ServerListener,
)

from google.protobuf import descriptor_pb2


# Network configuration (must match arduino_secrets.h)
# Note: Device now connects to WiFi as a client and gets IP via DHCP
# You must specify the device's IP address via --device-ip argument
# Check the Serial output or your router's DHCP table to find it
DEVICE_IP = None           # Will be set via command line argument
UDP_PORT = 7070            # Data logging port (CLIENT_PORT_UDP in arduino_secrets.h)
TCP_PORT = 8080            # REPL command port (CLIENT_PORT_TCP in arduino_secrets.h)


class ExampleListener(ServerListener):
    def __init__(self) -> None:
        # Map client id -> set of subscribed topics
        self.subscribers: dict[int, set[str]] = {}

    def has_subscribers(self) -> bool:
        return len(self.subscribers) > 0

    def on_subscribe(
        self,
        client: Client,
        channel: ChannelView,
    ) -> None:
        logging.info(f"Client {client} subscribed to channel {channel.topic}")
        self.subscribers.setdefault(client.id, set()).add(channel.topic)

    def on_unsubscribe(
        self,
        client: Client,
        channel: ChannelView,
    ) -> None:
        logging.info(f"Client {client} unsubscribed from channel {channel.topic}")
        self.subscribers[client.id].remove(channel.topic)
        if not self.subscribers[client.id]:
            del self.subscribers[client.id]

    def on_client_advertise(
        self,
        client: Client,
        channel: ClientChannel,
    ) -> None:
        logging.info(f"Client {client.id} advertised channel: {channel.id}")

    def on_message_data(
        self,
        client: Client,
        client_channel_id: int,
        data: bytes,
    ) -> None:
        logging.info(f"Message from client {client.id} on channel {client_channel_id}")

    def on_client_unadvertise(
        self,
        client: Client,
        client_channel_id: int,
    ) -> None:
        logging.info(f"Client {client.id} unadvertised channel: {client_channel_id}")


proto_fds = descriptor_pb2.FileDescriptorSet()
gen.Packet_pb2.DESCRIPTOR.CopyToProto(proto_fds.file.add())
packet_descriptor = gen.Packet_pb2.Packet.DESCRIPTOR # (basedpyright reportAttributeAccessIssue)


def make_trial_filename(trial_name: str, dirpath: Path = Path(".")) -> Path:
    """Generate a unique filename for a trial."""
    now = datetime.datetime.now()
    date_str = now.date().isoformat()
    time_str = f"{now.hour:02d}-{now.minute:02d}-{now.second:02d}"

    data_dir = dirpath / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    base = f"{trial_name}_{date_str}_{time_str}.mcap"
    candidate = data_dir / base

    trial = 1
    while candidate.exists():
        trial += 1
        candidate = data_dir / f"{trial_name}_{date_str}_{time_str}_{trial}.mcap"

    return candidate


class DeviceREPL:
    """TCP connection for sending commands to the device."""

    def __init__(self, host: str, port: int = TCP_PORT):
        self.host = host
        self.port = port
        self.sock = None
        self.connected = False
        self.receive_thread = None
        self.running = False

    def connect(self) -> bool:
        """Connect to the device TCP server."""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(5.0)
            self.sock.connect((self.host, self.port))
            self.sock.settimeout(0.1)
            self.connected = True
            self.running = True

            self.receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
            self.receive_thread.start()

            print(f"Connected to device REPL at {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"Failed to connect to device: {e}")
            return False

    def disconnect(self):
        """Disconnect from the device."""
        self.running = False
        self.connected = False
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
            self.sock = None
        if self.receive_thread:
            self.receive_thread.join(timeout=1.0)

    def _receive_loop(self):
        """Background thread to receive and print device responses."""
        while self.running:
            if not self.connected:
                # Try to reconnect
                time.sleep(1.0)
                if self._try_reconnect():
                    continue
                else:
                    continue

            try:
                data = self.sock.recv(1024)
                if data:
                    text = data.decode('utf-8', errors='replace').strip()
                    if text:  # Only print non-empty responses
                        print(f"[DEVICE] {text}")
                elif data == b'':
                    self.connected = False
                    print("Device disconnected, will attempt reconnection...")
            except socket.timeout:
                continue
            except (ConnectionResetError, BrokenPipeError, OSError) as e:
                if self.running:
                    print(f"Connection lost: {e}")
                    self.connected = False
                    print("Will attempt reconnection...")
            except Exception as e:
                if self.running:
                    print(f"Receive error: {e}")
                    self.connected = False

    def _try_reconnect(self) -> bool:
        """Attempt to reconnect to the device."""
        if not self.running:
            return False

        try:
            if self.sock:
                try:
                    self.sock.close()
                except:
                    pass

            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(5.0)
            self.sock.connect((self.host, self.port))
            self.sock.settimeout(0.1)
            self.connected = True
            print(f"Reconnected to device at {self.host}:{self.port}")
            return True
        except Exception as e:
            return False

    def send(self, command: str) -> bool:
        """Send a command to the device."""
        try:
            self.sock.sendall((command + "\n").encode('utf-8'))
            return True
        except Exception as e:
            print(f"Send error: {e}")
            self.connected = False
            return False

    def start_logging(self):
        return self.send("W")

    def stop_logging(self):
        return self.send("Q")

    def motor_state(self, state: int):
        if state < 0 or state > 4:
            print("Motor state must be 0-4")
            return False
        return self.send(f"M {state}")

    def motor_status(self):
        return self.send("S")

    def set_kp(self, value: float):
        return self.send(f"P {value}")

    def set_ki(self, value: float):
        return self.send(f"I {value}")

    def set_kd(self, value: float):
        return self.send(f"D {value}")

    def calibrate_adc(self):
        return self.send("J")

    def help(self):
        return self.send("H")


class TrialRecorder:
    """Manages recording data to MCAP and CSV files."""

    def __init__(self):
        self.mcap = None
        self.proto_chan = None
        self.recording = False
        self.current_filename = None
        self.csv_filename = None
        self.csv_classifier_filename = None  # Separate CSV for classifier packets
        self.csv_gait_filename = None  # Separate CSV for gait packets
        self.csv_train_filename = None  # Separate CSV for training packets
        self.t0_logger_ns = None
        self.t0_sample_time = None
        self.packet_count = 0
        self.last_sample_time = None
        self.dropped_packets = 0
        
        # CSV data storage
        self.csv_data = []  # IMUFrameHR packets
        self.csv_classifier_data = []  # IMUFrameClassifier packets
        self.csv_gait_data = []  # IMUFrameGait packets
        self.csv_train_data = []  # IMUFrameTrain packets

        # Network latency estimation
        self.latency_samples = []
        self.estimated_latency_ns = 0
        self.last_arduino_ms = None
        self.last_receive_ns = None
        
        # Rate tracking for periodic status updates
        self.last_status_time = None
        self.packets_since_last_status = 0
        self.STATUS_INTERVAL_SEC = 5.0

    def start_trial(self, trial_name: str) -> bool:
        """Start recording a new trial."""
        if self.recording:
            print(f"Already recording: {self.current_filename}")
            return False

        self.current_filename = make_trial_filename(trial_name)
        # CSV filename is same as MCAP but with .csv extension
        self.csv_filename = self.current_filename.with_suffix('.csv')
        self.mcap = foxglove.open_mcap(self.current_filename, allow_overwrite=True)

        # Create channel for this recording
        self.proto_chan = Channel(
            topic="/proto",
            message_encoding="protobuf",
            schema=Schema(
                name=f"{packet_descriptor.file.package}.{packet_descriptor.name}",
                encoding="protobuf",
                data=proto_fds.SerializeToString(),
            ),
        )

        self.recording = True
        self.t0_logger_ns = time.time_ns()
        self.t0_sample_time = None
        self.packet_count = 0
        self.csv_data = []  # Clear CSV data for new trial
        self.csv_classifier_data = []  # Clear classifier CSV data
        self.csv_gait_data = []  # Clear gait CSV data
        self.csv_train_data = []  # Clear training CSV data
        
        # Reset rate tracking
        self.last_status_time = time.time()
        self.packets_since_last_status = 0
        
        # Classifier and Gait CSV filenames
        self.csv_classifier_filename = self.current_filename.with_suffix('.classifier.csv')
        self.csv_gait_filename = self.current_filename.with_suffix('.gait.csv')
        self.csv_train_filename = self.current_filename.with_suffix('.train.csv')

        print(f"Started recording: {self.current_filename}")
        print(f"CSV will be saved to: {self.csv_filename}")
        print(f"Classifier CSV will be saved to: {self.csv_classifier_filename}")
        print(f"Gait CSV will be saved to: {self.csv_gait_filename}")
        print(f"Train CSV will be saved to: {self.csv_train_filename}")
        return True

    def stop_trial(self) -> bool:
        """Stop recording the current trial."""
        if not self.recording:
            print("Not currently recording")
            return False

        self.mcap.close()
        self.mcap = None
        self.proto_chan = None
        self.recording = False

        # Write CSV files
        self._write_csv()
        self._write_classifier_csv()
        self._write_gait_csv()
        self._write_train_csv()

        print(f"Stopped recording: {self.current_filename} ({self.packet_count} packets, {self.dropped_packets} dropped)")
        print(f"IMU CSV saved to: {self.csv_filename} ({len(self.csv_data)} samples)")
        print(f"Classifier CSV saved to: {self.csv_classifier_filename} ({len(self.csv_classifier_data)} samples)")
        print(f"Gait CSV saved to: {self.csv_gait_filename} ({len(self.csv_gait_data)} samples)")
        print(f"Train CSV saved to: {self.csv_train_filename} ({len(self.csv_train_data)} samples)")
        self.current_filename = None
        self.csv_filename = None
        self.csv_classifier_filename = None
        self.csv_gait_filename = None
        self.csv_train_filename = None
        self.last_sample_time = None
        self.dropped_packets = 0
        self.csv_data = []
        self.csv_classifier_data = []
        self.csv_gait_data = []
        self.csv_train_data = []
        self.last_status_time = None
        self.packets_since_last_status = 0
        return True

    def _write_csv(self):
        """Write IMUFrameHR data to CSV file."""
        if not self.csv_data:
            print("No IMU data to write to CSV")
            return

        # CSV header matching IMUFrameHR fields
        fieldnames = [
            'time_ns', 'pc_time', 'time_milis', 'sample_time_fine',
            'a_x', 'a_y', 'a_z',
            'w_x', 'w_y', 'w_z',
            'qw', 'qx', 'qy', 'qz',
            'dv_x', 'dv_y', 'dv_z',
            'dq_w', 'dq_x', 'dq_y', 'dq_z',
            'motor_state', 'adc_raw', 'adc_baseline'
        ]

        with open(self.csv_filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.csv_data)

    def _write_classifier_csv(self):
        """Write IMUFrameClassifier data to CSV file."""
        if not self.csv_classifier_data:
            print("No classifier data to write to CSV")
            return

        # CSV header matching IMUFrameClassifier fields (includes classifier output)
        fieldnames = [
            'time_ns', 'pc_time', 'time_milis', 'sample_time_fine',
            'a_x', 'a_y', 'a_z',
            'w_x', 'w_y', 'w_z',
            'vel_fwd', 'vel_lat', 'vel_z',
            'pos_fwd', 'pos_lat', 'pos_z',
            'w_sag', 'pitch',
            'motor_state', 'adc_raw', 'adc_baseline',
            'stance', 'foot_flat', 'shoe_statistic',
            'classifier_label', 'classifier_confidence', 'classifier_stable_label',
            'conf_dec', 'conf_dst', 'conf_idle', 'conf_inc', 'conf_lvl', 'conf_ust'

        ]

        with open(self.csv_classifier_filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.csv_classifier_data)

    def _write_gait_csv(self):
        """Write IMUFrameGait data to CSV file."""
        if not self.csv_gait_data:
            print("No gait data to write to CSV")
            return

        # CSV header matching IMUFrameGait fields
        fieldnames = [
            'time_ns', 'pc_time', 'time_milis', 'sample_time_fine',
            'a_x', 'a_y', 'a_z',
            'w_x', 'w_y', 'w_z',
            'vel_x', 'vel_y',
            'vel_fwd', 'vel_lat', 'vel_z',
            'pos_x', 'pos_y',
            'pos_fwd', 'pos_lat', 'pos_z',
            'w_sag', 'pitch',
            'walking_direction',
            'stance', 'foot_flat', 'shoe_statistic', 'step_count', 'gait_phase',
            'debounce_counter', 'raw_should_enter', 'raw_should_exit',
            'adc_raw', 'adc_baseline'
        ]

        with open(self.csv_gait_filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.csv_gait_data)

    def _write_train_csv(self):
        """Write IMUFrameTrain data to CSV file."""
        if not self.csv_train_data:
            print("No train data to write to CSV")
            return

        # CSV header matching IMUFrameTrain fields
        fieldnames = [
            'time_ns', 'pc_time', 'time_milis', 'sample_time_fine',
            'a_x', 'a_y', 'a_z',
            'w_x', 'w_y', 'w_z',
            'vel_x', 'vel_y',
            'vel_fwd', 'vel_lat', 'vel_z',
            'pos_x', 'pos_y',
            'pos_fwd', 'pos_lat', 'pos_z',
            'w_sag', 'pitch', 'adc_raw', 'adc_baseline',
            'stance', 'foot_flat', 'shoe_statistic'
        ]

        with open(self.csv_train_filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.csv_train_data)

    def log_packet(self, pkt, receive_time_ns: int | None = None):
        """Log a packet to the current trial.
        
        Handles both IMUFrameHR (standard mode) and IMUFrameClassifier (SEMIAUTO/AUTO mode)
        packets via the protobuf oneof 'payload' field.

        receive_time_ns: computer wall-clock time (time.time_ns()) when the UDP packet
            arrived, used for video sync with external cameras.
        """
        if not self.recording:
            return

        if receive_time_ns is None:
            receive_time_ns = time.time_ns()

        pc_time = datetime.datetime.fromtimestamp(receive_time_ns / 1e9).strftime('%Y-%m-%d %H:%M:%S.%f')

        sample_time = pkt.sample_time_fine

        if self.packet_count < 5:
            payload_type = pkt.WhichOneof('payload')
            print(f"Packet {self.packet_count}: type={payload_type}, sample_time_fine={sample_time}, time_milis={pkt.time_milis}")

        # Check for gaps (expected delta is 100 ticks = 10ms at 100Hz)
        if self.last_sample_time is not None:
            delta = sample_time - self.last_sample_time
            if delta > 150:  # More than 1.5x expected interval
                missed = (delta // 100) - 1
                self.dropped_packets += missed
                print(f"GAP: {missed} packets dropped (delta={delta} ticks)")
        self.last_sample_time = sample_time

        # Initialize on first packet
        if self.t0_sample_time is None:
            self.t0_sample_time = sample_time

        # Calculate time from MTi's 10kHz counter (handles wraparound)
        if sample_time >= self.t0_sample_time:
            delta_ticks = sample_time - self.t0_sample_time
        else:
            # Handle 32-bit wraparound
            delta_ticks = (0xFFFFFFFF - self.t0_sample_time) + sample_time + 1

        # Convert ticks to nanoseconds (10kHz = 100us per tick)
        delta_ns = delta_ticks * 100_000
        log_time_ns = self.t0_logger_ns + delta_ns

        self.proto_chan.log(pkt.SerializeToString(), log_time=log_time_ns)
        
        # Store data for CSV export based on packet type
        # The packet type is determined by the protobuf oneof 'payload' field
        if pkt.HasField('imu'):
            # Standard IMUFrameHR packet (used in manual/basic modes)
            imu = pkt.imu
            self.csv_data.append({
                'time_ns': log_time_ns,
                'pc_time': pc_time,
                'time_milis': pkt.time_milis,
                'sample_time_fine': sample_time,
                'a_x': imu.a_x, 'a_y': imu.a_y, 'a_z': imu.a_z,
                'w_x': imu.w_x, 'w_y': imu.w_y, 'w_z': imu.w_z,
                'qw': imu.qw, 'qx': imu.qx, 'qy': imu.qy, 'qz': imu.qz,
                'dv_x': imu.dv_x, 'dv_y': imu.dv_y, 'dv_z': imu.dv_z,
                'dq_w': imu.dq_w, 'dq_x': imu.dq_x, 'dq_y': imu.dq_y, 'dq_z': imu.dq_z,
                'motor_state': imu.motor_state,
                'adc_raw': imu.adc_raw,
                'adc_baseline': imu.adc_baseline
            })
        elif pkt.HasField('imu_classifier'):
            # IMUFrameClassifier packet (used in SEMIAUTO/AUTO modes)
            imu = pkt.imu_classifier
            self.csv_classifier_data.append({
                'time_ns': log_time_ns,
                'pc_time': pc_time,
                'time_milis': pkt.time_milis,
                'sample_time_fine': sample_time,
                'a_x': imu.a_x, 'a_y': imu.a_y, 'a_z': imu.a_z,
                'w_x': imu.w_x, 'w_y': imu.w_y, 'w_z': imu.w_z,
                'vel_fwd': imu.vel_fwd, 'vel_lat': imu.vel_lat, 'vel_z': imu.vel_z,
                'pos_fwd': imu.pos_fwd, 'pos_lat': imu.pos_lat, 'pos_z': imu.pos_z,
                'w_sag': imu.w_sag, 'pitch': imu.pitch,
                'motor_state': imu.motor_state,
                'adc_raw': imu.adc_raw,
                'adc_baseline': imu.adc_baseline,
                'stance': imu.stance,
                'foot_flat': imu.foot_flat,
                'shoe_statistic': imu.shoe_statistic,
                'classifier_label': imu.classifier_label,
                'classifier_confidence': imu.classifier_confidence,
                'classifier_stable_label': imu.classifier_stable_label,
                'conf_dec': imu.conf_dec,
                'conf_dst': imu.conf_dst,
                'conf_idle': imu.conf_idl,
                'conf_inc': imu.conf_inc,
                'conf_lvl': imu.conf_lvl,
                'conf_ust': imu.conf_ust
            })
        elif pkt.HasField('imu_gait'):
            # IMUFrameGait packet (used in gait filter mode)
            imu = pkt.imu_gait
            self.csv_gait_data.append({
                'time_ns': log_time_ns,
                'pc_time': pc_time,
                'time_milis': pkt.time_milis,
                'sample_time_fine': sample_time,
                'a_x': imu.a_x, 'a_y': imu.a_y, 'a_z': imu.a_z,
                'w_x': imu.w_x, 'w_y': imu.w_y, 'w_z': imu.w_z,
                'vel_x': imu.vel_x, 'vel_y': imu.vel_y,
                'vel_fwd': imu.vel_fwd, 'vel_lat': imu.vel_lat, 'vel_z': imu.vel_z,
                'pos_x': imu.pos_x, 'pos_y': imu.pos_y,
                'pos_fwd': imu.pos_fwd, 'pos_lat': imu.pos_lat, 'pos_z': imu.pos_z,
                'w_sag': imu.w_sag, 'pitch': imu.pitch,
                'walking_direction': imu.walking_direction,
                'stance': imu.stance,
                'foot_flat': imu.foot_flat,
                'shoe_statistic': imu.shoe_statistic,
                'step_count': imu.step_count,
                'gait_phase': imu.gait_phase,
                'debounce_counter': imu.debounce_counter,
                'raw_should_enter': imu.raw_should_enter,
                'raw_should_exit': imu.raw_should_exit,
                'adc_raw': imu.adc_raw,
                'adc_baseline': imu.adc_baseline
            })
        elif pkt.HasField('imu_train'):
            # IMUFrameTrain packet (used in gait training mode)
            imu = pkt.imu_train
            self.csv_train_data.append({
                'time_ns': log_time_ns,
                'pc_time': pc_time,
                'time_milis': pkt.time_milis,
                'sample_time_fine': sample_time,
                'a_x': imu.a_x, 'a_y': imu.a_y, 'a_z': imu.a_z,
                'w_x': imu.w_x, 'w_y': imu.w_y, 'w_z': imu.w_z,
                'vel_x': imu.vel_x, 'vel_y': imu.vel_y,
                'vel_fwd': imu.vel_fwd, 'vel_lat': imu.vel_lat, 'vel_z': imu.vel_z,
                'pos_x': imu.pos_x, 'pos_y': imu.pos_y,
                'pos_fwd': imu.pos_fwd, 'pos_lat': imu.pos_lat, 'pos_z': imu.pos_z,
                'w_sag': imu.w_sag, 'pitch': imu.pitch,
                'adc_raw': imu.adc_raw,
                'adc_baseline': imu.adc_baseline,
                'stance': imu.stance,
                'foot_flat': imu.foot_flat,
                'shoe_statistic': imu.shoe_statistic
            })
        
        self.packet_count += 1
        self.packets_since_last_status += 1
        
        # Periodic status update every STATUS_INTERVAL_SEC seconds
        now = time.time()
        elapsed = now - self.last_status_time
        if elapsed >= self.STATUS_INTERVAL_SEC:
            avg_rate = self.packets_since_last_status / elapsed
            imu_count = len(self.csv_data)
            classifier_count = len(self.csv_classifier_data)
            gait_count = len(self.csv_gait_data)
            train_count = len(self.csv_train_data)
            print(f"Logging Normal | Rate: {avg_rate:.1f} Hz | Total: {self.packet_count} pkts | IMU: {imu_count} | Classifier: {classifier_count} | Gait: {gait_count} | Train: {train_count} | Dropped: {self.dropped_packets}")
            self.last_status_time = now
            self.packets_since_last_status = 0


class InteractiveSession:
    """Main interactive session managing REPL and recording."""

    def __init__(self, device_ip: str = DEVICE_IP):
        self.repl = DeviceREPL(host=device_ip)
        self.recorder = TrialRecorder()
        self.running = False
        self.udp_sock = None

    def start(self):
        """Start the interactive session."""
        # Setup UDP socket with larger receive buffer
        self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)  # 1MB buffer
        self.udp_sock.bind(("0.0.0.0", UDP_PORT))
        self.udp_sock.setblocking(False)
        print(f"UDP socket bound to 0.0.0.0:{UDP_PORT}")

        # Connect to device (required)
        if not self.repl.connect():
            raise ConnectionError(f"Could not connect to device at {self.repl.host}:{self.repl.port}")

        # Start Foxglove server for live visualization
        foxglove.set_log_level(logging.WARNING)
        self.server = foxglove.start_server(
            port=8765,
            server_listener=ExampleListener(),
            capabilities=[Capability.ClientPublish],
            supported_encodings=["json"],
        )
        print("Foxglove server started on port 8765")

        self.running = True
        self._run_loop()

    def _run_loop(self):
        """Main loop handling UDP data and user input."""
        print("\n=== IMU Logger Interactive Session ===")
        print("Commands:")
        print("  start <name>  - Start recording trial with given name")
        print("  stop          - Stop current recording")
        print("  status        - Show recording status")
        print("  W/Q           - Start/stop device logging")
        print("  M <0-4>       - Set motor state")
        print("  S             - Get motor status")
        print("  P/I/D <val>   - Set PID gains")
        print("  J             - Calibrate ADC")
        print("  H             - Device help")
        print("  quit          - Exit session")
        print()

        # Use select for non-blocking input on Unix
        import sys
        if sys.platform != 'win32':
            import select as sel

        while self.running:
            # Drain ALL pending UDP datagrams before checking stdin.
            # At 100 Hz the device sends a packet every 10 ms; reading only
            # one datagram per iteration and then blocking on select(0.01)
            # causes the OS receive buffer to overflow and drop packets.
            while True:
                try:
                    data, addr = self.udp_sock.recvfrom(4096)
                    self._process_udp_data(data)
                except BlockingIOError:
                    break  # no more pending datagrams
                except Exception as e:
                    print(f"UDP error: {e}")
                    break

            # Check for user input (non-blocking)
            try:
                if sys.platform == 'win32':
                    import msvcrt
                    if msvcrt.kbhit():
                        cmd = input("> ").strip()
                        self._process_command(cmd)
                else:
                    rlist, _, _ = sel.select([sys.stdin], [], [], 0.01)
                    if rlist:
                        cmd = sys.stdin.readline().strip()
                        if cmd:
                            self._process_command(cmd)
                            print("> ", end="", flush=True)
            except EOFError:
                break
            except KeyboardInterrupt:
                break

        self._shutdown()

    def _process_udp_data(self, data: bytes):
        """Process incoming UDP data packets."""
        receive_time_ns = time.time_ns()  # Capture PC wall-clock time for video sync
        offset = 0
        count = 0
        while offset + 2 <= len(data):
            length, = struct.unpack_from('!H', data, offset)
            offset += 2

            if offset + length > len(data):
                print(f"WARNING: Truncated packet, need {length} bytes but only {len(data) - offset} remaining")
                break

            chunk = data[offset:offset + length]
            offset += length

            pkt = gen.Packet_pb2.Packet()
            pkt.ParseFromString(chunk)

            # Log to current trial if recording
            self.recorder.log_packet(pkt, receive_time_ns)
            count += 1

       # if count > 0:
            #print(f"Parsed {count} packets from {len(data)} bytes")

    def _process_command(self, cmd: str):
        """Process a user command."""
        if not cmd:
            return

        parts = cmd.split(maxsplit=1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        # Local commands
        if command == "start":
            if not args:
                print("Usage: start <trial_name>")
                return
            self.recorder.start_trial(args)
            # Note: send 'W' manually if device logging not already enabled

        elif command == "stop":
            self.recorder.stop_trial()

        elif command == "status":
            if self.recorder.recording:
                print(f"Recording: {self.recorder.current_filename}")
                print(f"Packets: {self.recorder.packet_count}")
            else:
                print("Not recording")
            print(f"Device connected: {self.repl.connected}")

        elif command in ("quit", "exit"):
            self.running = False

        # Device commands (pass through)
        elif command == "w":
            self.repl.start_logging()
        elif command == "q":
            self.repl.stop_logging()
        elif command == "m":
            try:
                state = int(args)
                self.repl.motor_state(state)
            except ValueError:
                print("Usage: M <0-4>")
        elif command == "s":
            self.repl.motor_status()
        elif command == "p":
            try:
                self.repl.set_kp(float(args))
            except ValueError:
                print("Usage: P <value>")
        elif command == "i":
            try:
                self.repl.set_ki(float(args))
            except ValueError:
                print("Usage: I <value>")
        elif command == "d":
            try:
                self.repl.set_kd(float(args))
            except ValueError:
                print("Usage: D <value>")
        elif command == "j":
            self.repl.calibrate_adc()
        elif command == "h":
            self.repl.help()
        else:
            # Send raw command to device
            self.repl.send(cmd)

    def _shutdown(self):
        """Clean shutdown."""
        print("\nShutting down...")
        if self.recorder.recording:
            self.repl.stop_logging()
            self.recorder.stop_trial()
        self.repl.disconnect()
        self.server.stop()
        if self.udp_sock:
            self.udp_sock.close()


def main():
    parser = argparse.ArgumentParser(
        description="IMU ESKF interactive logger with REPL support.",
        epilog="Example: python logger.py --device-ip 192.168.1.100"
    )
    parser.add_argument("--device-ip", type=str, required=True,
                        help="Device IP address (check Serial output or router DHCP table)")
    args = parser.parse_args()

    session = InteractiveSession(device_ip=args.device_ip)
    session.start()


if __name__ == "__main__":
    main()
