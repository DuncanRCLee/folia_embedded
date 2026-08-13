import argparse
import os
import queue
import socket
import struct
import sys
import threading
import time
import customtkinter as ctk
from PIL import Image

# Import the core classes you separated out
from logger_core import DeviceREPL, TrialRecorder, ExampleListener
import gen.Packet_pb2
import foxglove
from foxglove.websocket import Capability

UDP_PORT = 7070  

class StdoutRedirector:
    """Redirects standard print output to a thread-safe queue for the GUI."""
    def __init__(self, msg_queue):
        self.msg_queue = msg_queue

    def write(self, text):
        if text.strip():
            self.msg_queue.put(text.strip())

    def flush(self):
        pass


class AnkleLoggerGUI(ctk.CTk):
    def __init__(self, device_ip):
        super().__init__()

        self.device_ip = device_ip
        self.title("Prosthetic Ankle Operation Logger")
        self.geometry("1300x650")  # Widened to fit three panels comfortably
        
        self.repl = DeviceREPL(host=self.device_ip)
        self.recorder = TrialRecorder()
        self.running = False
        self.udp_sock = None
        
        self.msg_queue = queue.Queue()
        sys.stdout = StdoutRedirector(self.msg_queue)
        
        self.preload_images()
        self.setup_ui()
        self.protocol("WM_DELETE_WINDOW", self.end_session)

    def preload_images(self):
        """Preloads, crops, and scales all 5 state images into RAM on boot to prevent UI lag."""
        self.state_images = {}
        base_dir = os.path.dirname(os.path.abspath(__file__))
        img_dir = os.path.join(base_dir, "images")
        
        # Define the crop box: (left, upper, right, lower)
        crop_box = (159, 0, 1239, 1080)
        
        for i in range(5):
            # Updated to .png extension
            img_path = os.path.join(img_dir, f"state{i}.png")
            try:
                pil_img = Image.open(img_path)
                
                # Crop to 1080x1080 square
                cropped_img = pil_img.crop(crop_box)
                
                # Size set to 600x600 for a large, high-res display in the center panel
                self.state_images[i] = ctk.CTkImage(light_image=cropped_img, 
                                                    dark_image=cropped_img, 
                                                    size=(500, 500))
            except FileNotFoundError:
                print(f"Warning: Image {img_path} not found. State {i} will not display an image.")
                self.state_images[i] = None

    def setup_ui(self):
        """Constructs the three-panel GUI."""
        self.grid_columnconfigure(0, weight=1)  # Left panel (Controls)
        self.grid_columnconfigure(1, weight=1)  # Middle panel (Image)
        self.grid_columnconfigure(2, weight=2)  # Right panel (Messages)
        self.grid_rowconfigure(0, weight=1)

        # --- LEFT PANEL (Controls) ---
        left_panel = ctk.CTkScrollableFrame(self, corner_radius=10)
        left_panel.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        # 1. Motor Section
        motor_frame = ctk.CTkFrame(left_panel)
        motor_frame.pack(fill="x", padx=10, pady=10)
        ctk.CTkLabel(motor_frame, text="Motor Control", font=("Arial", 14, "bold")).pack(pady=5)
        
        btn_frame = ctk.CTkFrame(motor_frame, fg_color="transparent")
        btn_frame.pack(pady=5)
        for i in range(5):
            # Changed command to trigger the new image-swapping function
            ctk.CTkButton(btn_frame, text=str(i), width=35, 
                          command=lambda x=i: self.set_motor_state(x)).pack(side="left", padx=2)
            
        pid_frame = ctk.CTkFrame(motor_frame, fg_color="transparent")
        pid_frame.pack(pady=5)
        
        ctk.CTkLabel(pid_frame, text="P:").grid(row=0, column=0, padx=(5,2))
        self.entry_p = ctk.CTkEntry(pid_frame, width=45)
        self.entry_p.insert(0, "4.0")
        self.entry_p.grid(row=0, column=1)

        ctk.CTkLabel(pid_frame, text="I:").grid(row=0, column=2, padx=(5,2))
        self.entry_i = ctk.CTkEntry(pid_frame, width=45)
        self.entry_i.insert(0, "0.1")
        self.entry_i.grid(row=0, column=3)

        ctk.CTkLabel(pid_frame, text="D:").grid(row=0, column=4, padx=(5,2))
        self.entry_d = ctk.CTkEntry(pid_frame, width=45)
        self.entry_d.insert(0, "0.1")
        self.entry_d.grid(row=0, column=5)

        motor_action_frame = ctk.CTkFrame(motor_frame, fg_color="transparent")
        motor_action_frame.pack(pady=5)
        
        ctk.CTkButton(motor_action_frame, text="Set PID", width=100, 
                      command=self.set_pid).pack(side="left", padx=5)
        
        ctk.CTkButton(motor_action_frame, text="Motor Status", width=100, 
                      command=self.repl.motor_status).pack(side="left", padx=5)

        # 2. ADC Section
        adc_frame = ctk.CTkFrame(left_panel)
        adc_frame.pack(fill="x", padx=10, pady=10)
        ctk.CTkLabel(adc_frame, text="ADC", font=("Arial", 14, "bold")).pack(pady=5)
        ctk.CTkButton(adc_frame, text="Calibrate ADC", command=self.repl.calibrate_adc).pack(pady=5)

        # 3. Recording Section
        rec_frame = ctk.CTkFrame(left_panel)
        rec_frame.pack(fill="x", padx=10, pady=10)
        ctk.CTkLabel(rec_frame, text="Recording", font=("Arial", 14, "bold")).pack(pady=5)
        
        enable_frame = ctk.CTkFrame(rec_frame, fg_color="transparent")
        enable_frame.pack(pady=5)
        ctk.CTkButton(enable_frame, text="Enable", width=80, 
                      command=lambda: self.toggle_recording(True)).pack(side="left", padx=5)
        ctk.CTkButton(enable_frame, text="Disable", width=80, 
                      command=lambda: self.toggle_recording(False)).pack(side="left", padx=5)

        self.entry_rec_name = ctk.CTkEntry(rec_frame, placeholder_text="Recording Name")
        self.entry_rec_name.pack(pady=5, fill="x", padx=10)

        start_stop_frame = ctk.CTkFrame(rec_frame, fg_color="transparent")
        start_stop_frame.pack(pady=5)
        self.btn_start = ctk.CTkButton(start_stop_frame, text="Start", width=80, 
                                       state="disabled", command=self.start_recording)
        self.btn_start.pack(side="left", padx=5)
        
        self.btn_stop = ctk.CTkButton(start_stop_frame, text="Stop", width=80, 
                                      state="disabled", command=self.stop_recording)
        self.btn_stop.pack(side="left", padx=5)
        
        ctk.CTkButton(rec_frame, text="Recording Status", command=self.print_status).pack(pady=5)

        # 4. Help/Exit Section
        sys_frame = ctk.CTkFrame(left_panel)
        sys_frame.pack(fill="x", padx=10, pady=10)
        ctk.CTkLabel(sys_frame, text="System", font=("Arial", 14, "bold")).pack(pady=5)
        ctk.CTkButton(sys_frame, text="GUI Help", command=self.show_help).pack(pady=5)
        ctk.CTkButton(sys_frame, text="End Session", fg_color="#c0392b", hover_color="#922b21", 
                      command=self.end_session).pack(pady=5)


        # --- MIDDLE PANEL (Image Viewer) ---
        mid_panel = ctk.CTkFrame(self, corner_radius=10)
        mid_panel.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        ctk.CTkLabel(mid_panel, text="Folia State", font=("Arial", 14, "bold")).pack(pady=5)
        
        # Create an empty label to hold the image
        self.image_label = ctk.CTkLabel(mid_panel, text="")
        self.image_label.pack(expand=True, padx=10, pady=10)
        
        # Load the default State 0 image on boot
        if self.state_images.get(0):
            self.image_label.configure(image=self.state_images[0])


        # --- RIGHT PANEL (Message Window) ---
        right_panel = ctk.CTkFrame(self, corner_radius=10)
        right_panel.grid(row=0, column=2, padx=10, pady=10, sticky="nsew")
        
        ctk.CTkLabel(right_panel, text="System Messages", font=("Arial", 14, "bold")).pack(pady=5)
        self.textbox = ctk.CTkTextbox(right_panel, state="disabled", wrap="word")
        self.textbox.pack(fill="both", expand=True, padx=10, pady=10)

    # --- GUI Event Handlers ---
    def set_motor_state(self, state):
        """Sends the motor state to the device and updates the center panel image."""
        self.repl.motor_state(state)
        if self.state_images.get(state):
            self.image_label.configure(image=self.state_images[state])

    def set_pid(self):
        try:
            self.repl.set_kp(float(self.entry_p.get()))
            time.sleep(0.05)
            self.repl.set_ki(float(self.entry_i.get()))
            time.sleep(0.05)
            self.repl.set_kd(float(self.entry_d.get()))
            print("PID values sent to device.")
        except ValueError:
            print("Error: P, I, and D values must be valid numbers.")

    def update_pid_fields(self, p, i, d):
        self.after(0, self._apply_pid_fields, p, i, d)

    def _apply_pid_fields(self, p, i, d):
        self.entry_p.delete(0, 'end')
        self.entry_p.insert(0, p)
        
        self.entry_i.delete(0, 'end')
        self.entry_i.insert(0, i)
        
        self.entry_d.delete(0, 'end')
        self.entry_d.insert(0, d)
        print("GUI fields successfully synced with device PID.")

    def toggle_recording(self, enable):
        """Enables or disables the recording buttons intelligently based on current status."""
        if enable:
            if not self.recorder.recording:
                self.btn_start.configure(state="normal")
                self.btn_stop.configure(state="disabled")
            else:
                # If we are already recording, just re-enable the stop button
                self.btn_start.configure(state="disabled")
                self.btn_stop.configure(state="normal")
            print("Recording controls enabled.")
        else:
            self.btn_start.configure(state="disabled")
            self.btn_stop.configure(state="disabled")
            print("Recording controls disabled.")

    def start_recording(self):
        name = self.entry_rec_name.get().strip()
        if not name:
            print("Error: Please enter a recording name.")
            return
        
        # Only swap button states if the trial actually successfully started
        if self.recorder.start_trial(name):
            self.repl.start_logging() 
            self.btn_start.configure(state="disabled")
            self.btn_stop.configure(state="normal")

    def stop_recording(self):
        self.repl.stop_logging()
        
        # Only swap button states if the trial actually stopped successfully
        if self.recorder.stop_trial():
            self.btn_start.configure(state="normal")
            self.btn_stop.configure(state="disabled")

    def print_status(self):
        if self.recorder.recording:
            print(f"Recording Active: {self.recorder.current_filename}")
            print(f"Packets collected: {self.recorder.packet_count}")
        else:
            print("Recording Inactive.")
        print(f"Device connected: {self.repl.connected}")

    def show_help(self):
        help_text = (
            "--- GUI Functionality Guide ---\n"
            "MOTOR CONTROL:\n"
            "  • Buttons 0-4: Immediately set the active state of the ankle motor.\n"
            "  • PID: Enter desired values and click 'Set PID' to push tuning to the device.\n\n"
            "ADC / STRAIN GAUGE:\n"
            "  • Calibrate: Zeroes out the baseline of the ADS1220.\n\n"
            "RECORDING:\n"
            "  • Enable: Unlocks the start/stop buttons to prevent accidental clicks.\n"
            "  • Start: Creates a new .mcap and .csv files in the ./data/ directory.\n"
            "  • Status: Prints current packet counts and file paths.\n"
            "-------------------------------"
        )
        print(help_text)

    # --- Background Processing ---
    def process_message_queue(self):
        messages = []
        while not self.msg_queue.empty():
            try:
                messages.append(self.msg_queue.get_nowait())
            except queue.Empty:
                break
        
        if messages:
            self.textbox.configure(state="normal")
            for msg in messages:
                self.textbox.insert("end", msg + "\n")
            self.textbox.insert("end", "\n")
            self.textbox.see("end")
            self.textbox.configure(state="disabled")
            
        if self.running:
            self.after(50, self.process_message_queue)

    def udp_listen_loop(self):
        while self.running:
            try:
                data, addr = self.udp_sock.recvfrom(4096)
                receive_time_ns = time.time_ns()
                offset = 0
                while offset + 2 <= len(data):
                    length, = struct.unpack_from('!H', data, offset)
                    offset += 2
                    if offset + length > len(data):
                        break
                    
                    chunk = data[offset:offset + length]
                    offset += length
                    
                    pkt = gen.Packet_pb2.Packet()
                    pkt.ParseFromString(chunk)
                    
                    self.recorder.log_packet(pkt, receive_time_ns)
            except BlockingIOError:
                time.sleep(0.001)
            except OSError as e:
                if self.running:
                    print(f"UDP Socket Error: {e}")
                break
            except Exception as e:
                print(f"Error processing packet: {e}")

    # --- Session Management ---
    def start_session(self):
        self.running = True
        self.repl.pid_callback = self.update_pid_fields
        
        self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
        self.udp_sock.bind(("0.0.0.0", UDP_PORT))
        self.udp_sock.setblocking(False)
        print(f"UDP listening on 0.0.0.0:{UDP_PORT}")

        if not self.repl.connect():
            print(f"WARNING: Could not connect to device at {self.device_ip}")
        else:
            self.repl.request_pid()

        self.server = foxglove.start_server(
            port=8765,
            server_listener=ExampleListener(),
            capabilities=[Capability.ClientPublish],
            supported_encodings=["json"],
        )
        print("Foxglove visualization server started on port 8765")

        self.udp_thread = threading.Thread(target=self.udp_listen_loop, daemon=True)
        self.udp_thread.start()
        self.after(100, self.process_message_queue)

    def end_session(self):
        print("Safely shutting down session and hardware...")
        self.running = False
        
        if self.repl.connected:
            self.repl.motor_state(0)  
            self.repl.stop_logging()  
            time.sleep(0.1)           
        
        if self.recorder.recording:
            self.recorder.stop_trial()
            
        self.repl.disconnect()
        self.server.stop()
        if self.udp_sock:
            self.udp_sock.close()
            
        sys.stdout = sys.__stdout__
        self.destroy()


def main():
    parser = argparse.ArgumentParser(description="Ankle Logger GUI")
    parser.add_argument("--device-ip", type=str, default="192.168.4.1", 
                        help="Device IP address (defaults to 192.168.4.1)")
    args = parser.parse_args()

    ctk.set_appearance_mode("Dark")
    ctk.set_default_color_theme("blue")
    
    app = AnkleLoggerGUI(device_ip=args.device_ip)
    app.after(100, app.start_session)
    app.mainloop()

if __name__ == "__main__":
    main()