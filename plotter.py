import serial
import threading
import collections
import numpy as np
from scipy.signal import butter, iirnotch, lfilter
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# PLOTTER
# This python script is a rough prototype that pulls a raw ecg trace over serial.
# It's designed to test processing techniques before they're ported over to the mcu.

# --- CONFIGURATION ---
SERIAL_PORT = '/dev/ttyACM0'         # Change to your exact port ('/dev/ttyUSB0' on Linux)
BAUD_RATE = 2000000           # Match your ESP-IDF configuration
ECG_WINDOW = 2000            # Show 1 second of live wave data on screen
TREND_WINDOW = 30            # Store the last 30 beats for metrics calculation
# ---------------------

# Ring buffers for the data used
ecg_buffer = collections.deque(maxlen=ECG_WINDOW)
bpm_buffer = collections.deque(maxlen=TREND_WINDOW)
hrv_buffer = collections.deque(maxlen=TREND_WINDOW)
rr_buffer = collections.deque(maxlen=TREND_WINDOW)
is_running = True

class AdvancedECGProcessor:
    def __init__(self, fs=1000.0):
        self.fs = fs
        
        # 1. Digital Filter Stages (0.5Hz - 40Hz Bandpass + 60Hz Notch)
        b_hp, a_hp = butter(1, 0.5, btype='high', fs=self.fs)
        self.hp_state = np.zeros(max(len(b_hp), len(a_hp)) - 1)
        self.b_hp, self.a_hp = b_hp, a_hp

        b_notch, a_notch = iirnotch(60.0, 30.0, fs=self.fs)
        self.notch_state = np.zeros(max(len(b_notch), len(a_notch)) - 1)
        self.b_notch, self.a_notch = b_notch, a_notch

        b_lp, a_lp = butter(2, 40.0, btype='low', fs=self.fs)
        self.lp_state = np.zeros(max(len(b_lp), len(a_lp)) - 1)
        self.b_lp, self.a_lp = b_lp, a_lp

        # R-Peak & Metric Tracking Buffers
        self.threshold = 1000.0  
        self.cooldown_samples = int(0.3 * self.fs)
        self.cooldown_counter = 0
        self.sample_counter = 0
        
        self.r_peak_times = []      # Timestamps of recent peaks (in seconds)
        self.r_peak_amplitudes = [] # Peak heights for EDR estimation
        self.rr_intervals = []      # Inter-beat intervals (in milliseconds)

    def process_sample(self, raw_sample):
        self.sample_counter += 1
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1

        # Feed the cascaded filter chain
        clean, self.hp_state = lfilter(self.b_hp, self.a_hp, [raw_sample], zi=self.hp_state)
        clean, self.notch_state = lfilter(self.b_notch, self.a_notch, clean, zi=self.notch_state)
        clean, self.lp_state = lfilter(self.b_lp, self.a_lp, clean, zi=self.lp_state)
        filtered_val = float(clean[0])

        metrics = {"bpm": None, "hrv": None, "rr": None}

        # Peak Detection Check
        if filtered_val > self.threshold and self.cooldown_counter == 0:
            current_time_sec = self.sample_counter / self.fs
            
            if len(self.r_peak_times) > 0:
                last_time_sec = self.r_peak_times[-1]
                ibi_ms = (current_time_sec - last_time_sec) * 1000.0 # Interval in ms
                
                # Filter out impossible biometric intervals (40-200 BPM equivalent)
                if 300.0 <= ibi_ms <= 1500.0:
                    self.r_peak_times.append(current_time_sec)
                    self.r_peak_amplitudes.append(filtered_val)
                    self.rr_intervals.append(ibi_ms)
                    
                    # Keep local analytics tracking slices lean
                    if len(self.r_peak_times) > 40:
                        self.r_peak_times.pop(0)
                        self.r_peak_amplitudes.pop(0)
                        self.rr_intervals.pop(0)

                    # --- CALCULATE BIOMETRICS ---
                    # 1. Heart Rate (BPM)
                    metrics["bpm"] = 60000.0 / ibi_ms
                    
                    # 2. HRV (RMSSD calculation over last 5-10 beats)
                    if len(self.rr_intervals) >= 5:
                        diffs = np.diff(self.rr_intervals)
                        metrics["hrv"] = np.sqrt(np.mean(diffs ** 2))
                    
                    # 3. Respiration Rate (EDR calculation)
                    # We compute respiration frequency by measuring the cycling rate of the R-peak heights
                    if len(self.r_peak_times) >= 12:
                        times = np.array(self.r_peak_times)
                        amps = np.array(self.r_peak_amplitudes)
                        
                        # Detrend the amplitude variance to isolate breathing movements
                        amps_detrended = amps - np.mean(amps)
                        intervals = np.diff(times)
                        avg_ibi_sec = np.mean(intervals) if len(intervals) > 0 else 0.8
                        
                        # Count zero-crossings of the amplitude deviations to estimate breath cycles [4]
                        zero_crossings = np.where(np.diff(np.sign(amps_detrended)))[0]
                        total_time = times[-1] - times[0]
                        
                        if total_time > 0:
                            breaths = len(zero_crossings) / 2.0
                            calculated_rr = (breaths / total_time) * 60.0
                            if 6 <= calculated_rr <= 35: # Bound to normal human respiration scales [5]
                                metrics["rr"] = calculated_rr

            else:
                self.r_peak_times.append(current_time_sec)
                self.r_peak_amplitudes.append(filtered_val)

            self.cooldown_counter = self.cooldown_samples

        # Adaptive peak decay
        self.threshold = self.threshold * 0.995 + (filtered_val * 0.005 if filtered_val > 0 else 0)
        self.threshold = max(self.threshold, 200.0)

        return filtered_val, metrics

def serial_reader_thread():
    global is_running
    processor = AdvancedECGProcessor(fs=1000.0)
    
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"Reading telemetry from {SERIAL_PORT}...")
        ser.reset_input_buffer()
        
        while is_running:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line:
                try:
                    raw_val = int(line)
                    filtered_val, metrics = processor.process_sample(raw_val)
                    
                    ecg_buffer.append(filtered_val)
                    if metrics["bpm"] is not None:
                        bpm_buffer.append(metrics["bpm"])
                    if metrics["hrv"] is not None:
                        hrv_buffer.append(metrics["hrv"])
                    if metrics["rr"] is not None:
                        rr_buffer.append(metrics["rr"])
                except ValueError:
                    continue
    except serial.SerialException as e:
        print(f"Serial Link Closed: {e}")
    finally:
        is_running = False

# Run background thread
threading.Thread(target=serial_reader_thread, daemon=True).start()

# --- MATPLOTLIB VISUALIZATION LAYOUT ---
fig, (ax_ecg, ax_bpm, ax_stats) = plt.subplots(3, 1, figsize=(10, 8))
fig.suptitle("AD8232 Advanced Cardio-Pulmonary Diagnostic Dashboard", fontsize=13, fontweight='bold')

# 1. Top Panel: Wave
line_ecg, = ax_ecg.plot([], [], lw=1.2, color='tab:red')
ax_ecg.set_title("Filtered Live ECG Waveform")
ax_ecg.set_ylim(-1000, 2000)
ax_ecg.grid(True, linestyle=':', alpha=0.5)

# 2. Middle Panel: Heart Rate Trend
line_bpm, = ax_bpm.plot([], [], lw=1.8, color='tab:blue', marker='o', markersize=3)
ax_bpm.set_title("Heart Rate Trend (BPM)")
ax_bpm.set_ylim(40, 160)
ax_bpm.grid(True, linestyle=':', alpha=0.5)

# 3. Bottom Panel: Dual HRV & Respiration Axis
line_hrv, = ax_stats.plot([], [], lw=1.5, color='tab:purple', marker='s', markersize=3, label='HRV (RMSSD)')
ax_stats.set_title("Autonomic Biometrics History")
ax_stats.set_ylabel("HRV (ms)", color='tab:purple')
ax_stats.set_ylim(0, 150)
ax_stats.tick_params(axis='y', labelcolor='tab:purple')

ax_rr = ax_stats.twinx() # Share same X axis for respiratory plots
line_rr, = ax_rr.plot([], [], lw=1.5, color='tab:green', marker='^', markersize=3, label='Respiration Rate')
ax_rr.set_ylabel("Breaths / Min (BRPM)", color='tab:green')
ax_rr.set_ylim(8, 50)
ax_rr.tick_params(axis='y', labelcolor='tab:green')
ax_stats.grid(True, linestyle=':', alpha=0.5)

# Dashboard HUD Overlay
hud_text = fig.text(0.02, 0.02, "HR: -- BPM  |  HRV: -- ms  |  Resp: -- BRPM", 
                    fontsize=11, fontweight='bold', family='monospace',
                    bbox=dict(facecolor='lightgray', alpha=0.5, edgecolor='none'))

plt.tight_layout()

def update_plot(frame):
    current_ecg = list(ecg_buffer)
    current_bpm = list(bpm_buffer)
    current_hrv = list(hrv_buffer)
    current_rr = list(rr_buffer)

    # Refresh Top Axis
    line_ecg.set_data(np.arange(len(current_ecg)), current_ecg)
    ax_ecg.set_xlim(0, max(len(current_ecg), 1))

    # Refresh Middle Axis
    line_bpm.set_data(np.arange(len(current_bpm)), current_bpm)
    ax_bpm.set_xlim(0, max(len(current_bpm), 1))

    # Refresh Bottom Shared Axes
    line_hrv.set_data(np.arange(len(current_hrv)), current_hrv)
    line_rr.set_data(np.arange(len(current_rr)), current_rr)
    ax_stats.set_xlim(0, max(max(len(current_hrv), len(current_rr)), 1))

    # Update Text Stats Box
    hr_str = f"{current_bpm[-1]:.1f}" if current_bpm else "--"
    hrv_str = f"{current_hrv[-1]:.1f}" if current_hrv else "--"
    rr_str = f"{current_rr[-1]:.1f}" if current_rr else "--"
    hud_text.set_text(f"Current Heart Rate: {hr_str} BPM  |  HRV: {hrv_str} ms  |  Respiration: {rr_str} BRPM")

    return line_ecg, line_bpm, line_hrv, line_rr, hud_text

ani = animation.FuncAnimation(fig, update_plot, interval=33, blit=False, cache_frame_data=False)

try:
    plt.subplots_adjust(bottom=0.08)
    plt.show()
except KeyboardInterrupt:
    pass
finally:
    is_running = False
