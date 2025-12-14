# alikay_h_heart_rate_monitor_final_2025.py


import cv2
import numpy as np
import time
from scipy.signal import butter, filtfilt, find_peaks
import matplotlib.pyplot as plt

FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
GREEN_CHANNEL = 1
BUFFER_SIZE = 450
FS = 30  # تقریبی FPS

def bandpass_filter(data, lowcut=0.7, highcut=4.0, fs=FS, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

class HeartRateMonitor:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.buffer = np.zeros(BUFFER_SIZE)
        self.index = 0
        self.bpm = 0

        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        self.ax1.set_title("تصویر زنده وب‌کم")
        self.ax1.axis('off')
        self.img_plot = self.ax1.imshow(np.zeros((480,640,3), dtype=np.uint8))

        self.x_data = np.arange(BUFFER_SIZE)
        self.line_raw, = self.ax2.plot(self.x_data, self.buffer, 'b-', label="سیگنال خام", alpha=0.5)
        self.line_filtered, = self.ax2.plot(self.x_data, np.zeros(BUFFER_SIZE), 'g-', lw=2, label="سیگنال فیلترشده")
        self.ax2.set_ylim(50, 200)
        self.ax2.set_title(f"ضربان قلب: {int(self.bpm)} BPM")
        self.ax2.legend()

    def run(self):
        print("وب‌کم باز شد — صورتت رو ثابت جلوی دوربین بگیر، نور ثابت باشه 😈")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = FACE_CASCADE.detectMultiScale(gray, 1.3, 5)

            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                forehead = frame[y:y + h//2, x:x + w]
                green_mean = np.mean(forehead[:, :, GREEN_CHANNEL])

                self.buffer[self.index] = green_mean
                self.index = (self.index + 1) % BUFFER_SIZE

                # آپدیت گراف خام همیشه
                current_len = min(self.index + 1, BUFFER_SIZE)
                self.line_raw.set_data(self.x_data[:current_len], self.buffer[:current_len])

                # وقتی بافر پر شد، تحلیل دقیق
                if self.index == 0 and np.std(self.buffer) > 2:
                    filtered = bandpass_filter(self.buffer)
                    self.line_filtered.set_data(self.x_data, filtered)

                    peaks, _ = find_peaks(filtered, height=np.mean(filtered), distance=FS*0.6)
                    if len(peaks) > 3:
                        intervals = np.diff(peaks) / FS
                        self.bpm = 60 / np.mean(intervals)
                        self.bpm = round(max(40, min(180, self.bpm)))
                    else:
                        self.bpm = 0

                self.ax2.set_title(f"ضربان قلب: {int(self.bpm)} BPM")

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"BPM: {int(self.bpm)}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.img_plot.set_data(rgb_frame)
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

            # خروج با کلیک روی پنجره یا q
            if plt.waitforbuttonpress(timeout=0.001):
                break

        self.cap.release()
        plt.close('all')

if __name__ == "__main__":
    monitor = HeartRateMonitor()
    monitor.run()