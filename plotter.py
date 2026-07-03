import serial
import threading
import collections
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- CONFIGURATION ---
SERIAL_PORT = '/dev/ttyACM0'       # Update this to your ESP32-S3 port
BAUD_RATE = 2000000         # Match your ESP-IDF sdkconfig baud rate
ECG_WINDOW = 2000            # 1000 samples = 1 second of live wave data
BPM_WINDOW = 30              # Keep track of the last 30 detected heartbeats
# ---------------------

# Fast thread-safe ring buffer to hold the plotting window data
ecg_buffer = collections.deque([0] * ECG_WINDOW, maxlen=ECG_WINDOW)
bpm_buffer = collections.deque([0] * BPM_WINDOW, maxlen=BPM_WINDOW)

is_running = True
import numpy as np
from scipy.signal import butter, iirnotch, lfilter

class ECGProcessor:
    def __init__(self, fs=1000.0):
        self.fs = fs  # Sampling rate: 1000 Hz
        
        # 1. Baseline Wander Filter: High-pass at 0.5 Hz (removes breathing drift)
        b_hp, a_hp = butter(1, 0.5, btype='high', fs=self.fs)
        self.hp_state = np.zeros(max(len(b_hp), len(a_hp)) - 1)
        self.b_hp, self.a_hp = b_hp, a_hp

        # 2. Powerline Interference Filter: Notch at 60 Hz (or 50 Hz depending on your region)
        b_notch, a_notch = iirnotch(60.0, 30.0, fs=self.fs)
        self.notch_state = np.zeros(max(len(b_notch), len(a_notch)) - 1)
        self.b_notch, self.a_notch = b_notch, a_notch

        # 3. High-Frequency Noise Filter: Low-pass at 40 Hz (removes muscle twitch fuzz)
        b_lp, a_lp = butter(2, 40.0, btype='low', fs=self.fs)
        self.lp_state = np.zeros(max(len(b_lp), len(a_lp)) - 1)
        self.b_lp, self.a_lp = b_lp, a_lp

        # R-Peak / Heart Rate Tracking states
        self.threshold = 1500.0  # Dynamic threshold tracker
        self.cooldown_samples = int(0.3 * self.fs)  # 300ms refractory period between beats
        self.cooldown_counter = 0
        self.last_peak_sample = 0
        self.sample_counter = 0

    def process_sample(self, raw_sample):
        """Processes a single incoming sample in real-time using streaming filter states."""
        self.sample_counter += 1
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1

        # Apply cascaded filters using lfilter_zi states (mimics MCU rolling buffers)
        clean, self.hp_state = lfilter(self.b_hp, self.a_hp, [raw_sample], zi=self.hp_state)
        clean, self.notch_state = lfilter(self.b_notch, self.a_notch, clean, zi=self.notch_state)
        clean, self.lp_state = lfilter(self.b_lp, self.a_lp, clean, zi=self.lp_state)
        
        filtered_val = clean[0]

        # 4. R-Peak Detection (Pan-Tompkins inspired thresholding)
        bpm = None
        # Look for a sharp rising edge that clears the dynamic threshold
        if filtered_val > self.threshold and self.cooldown_counter == 0:
            if self.last_peak_sample > 0:
                # Calculate duration between current peak and last peak
                samples_between = self.sample_counter - self.last_peak_sample
                seconds_between = samples_between / self.fs
                bpm = 60.0 / seconds_between
            
            self.last_peak_sample = self.sample_counter
            self.cooldown_counter = self.cooldown_samples

        # Slowly decay threshold to adapt to varying ECG amplitudes
        self.threshold = self.threshold * 0.995 + (filtered_val * 0.005 if filtered_val > 0 else 0)
        # Prevent threshold from dropping to zero during noise
        self.threshold = max(self.threshold, 200.0) 

        return filtered_val, bpm

def serial_reader_thread():
    """Background thread to read serial data continuously without blocking the UI."""
    global is_running
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"Connected to {SERIAL_PORT} at {BAUD_RATE} baud.")
        
        # Flush initial junk bytes out of the buffer
        ser.reset_input_buffer()
        processor = ECGProcessor(fs=1000)
        
        while is_running:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line:
                try:
                    # Parse the integer printed by your ESP32-S3
                    val = int(line)
                    filtered_val, bpm = processor.process_sample(val)
                    ecg_buffer.append(filtered_val)
                    if bpm is not None:
                        bpm_buffer.append(bpm)

                except ValueError:
                    # Ignore lines that aren't pure integers (e.g., boot messages)
                    continue
    except serial.SerialException as e:
        print(f"Serial Error: {e}")
    finally:
        print("Serial reader thread stopped.")

# Start the background data collector
reader_thread = threading.Thread(target=serial_reader_thread, daemon=True)
reader_thread.start()

# 4. Generate Matplotlib Subplots Layout (2 Rows, 1 Column)
fig, (ax_ecg, ax_bpm) = plt.subplots(2, 1, figsize=(10, 6), sharex=False)
fig.suptitle("AD8232 ECG Real-Time Analytics Pipeline", fontsize=14, fontweight='bold')

# Setup Top Canvas: Filtered Wave
line_ecg, = ax_ecg.plot(list(ecg_buffer), lw=1.5, color='tab:red')
ax_ecg.set_title("Filtered ECG Signal (1 kHz Continuous)")
ax_ecg.set_ylabel("Amplitude")
ax_ecg.set_ylim(-1000, 2000) # Centered around high-pass filter 0-line
ax_ecg.grid(True, linestyle=':', alpha=0.6)

# Setup Bottom Canvas: Running BPM Trace
line_bpm, = ax_bpm.plot(list(bpm_buffer), lw=2.0, color='tab:blue', marker='o', markersize=4)
text_bpm = ax_bpm.text(0.02, 0.85, 'Current HR: -- BPM', transform=ax_bpm.transAxes, 
                        fontsize=12, fontweight='bold', color='tab:blue',
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
ax_bpm.set_title("Heart Rate Trend (Beats Per Minute)")
ax_bpm.set_xlabel("Recent Detected Beats")
ax_bpm.set_ylabel("BPM")
ax_bpm.set_ylim(40, 160) # Standard physiological resting/exercise boundary
ax_bpm.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()

def update_plot(frame):
    # Snapshot current buffer content
    current_ecg = list(ecg_buffer)
    current_bpm = list(bpm_buffer)
    
    # 5. Refresh Waveform Line
    line_ecg.set_xdata(np.arange(len(current_ecg)))
    line_ecg.set_ydata(current_ecg)
    ax_ecg.set_xlim(0, max(len(current_ecg), 1))
    
    # 6. Refresh Trend Scatter Line
    if current_bpm:
        line_bpm.set_xdata(np.arange(len(current_bpm)))
        line_bpm.set_ydata(current_bpm)
        ax_bpm.set_xlim(0, max(len(current_bpm), 1))
        # Update text overlay with the latest calculated heartbeat value
        text_bpm.set_text(f"Current HR: {current_bpm[-1]:.1f} BPM")
    
    return line_ecg, line_bpm, text_bpm

# Run visualization cycle at ~30 FPS
ani = animation.FuncAnimation(fig, update_plot, interval=33, blit=False, cache_frame_data=False)

try:
    plt.show()
except KeyboardInterrupt:
    pass
finally:
    is_running = False
    print("Exiting plotter application.")
