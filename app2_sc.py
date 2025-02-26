import cv2
import time
import threading
import queue
import os
import csv
from dotenv import load_dotenv
import signal
from ultralytics import YOLO
import gc
import psutil
from contextlib import contextmanager
def monitor_memory():
    """Get current memory usage"""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)

# Load environment variables
load_dotenv()

# Graceful shutdown flag
shutdown_event = threading.Event()

class MemoryMonitor:
    def __init__(self, threshold_mb=500):
        self.threshold_mb = threshold_mb
        self.process = psutil.Process()
        self.lock = threading.Lock()
        
    def get_memory_usage(self):
        with self.lock:
            return self.process.memory_info().rss / (1024 * 1024)
            
    def should_cleanup(self):
        return self.get_memory_usage() > self.threshold_mb

class VideoCapture:
    def __init__(self, name, max_queue_size=2):
        self.cap = cv2.VideoCapture(name)
        self.q = queue.Queue(maxsize=max_queue_size)  # Limit queue size
        self.running = True
        self.t = threading.Thread(target=self._reader)
        self.t.daemon = True
        self.t.start()

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            # Clear queue if full
            while self.q.full():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            self.q.put((frame.copy(), ret))  # Store a copy to prevent reference issues
            
    def read(self):
        try:
            frame, ret = self.q.get(timeout=1.0)
            return frame, ret
        except queue.Empty:
            return None, False

    def stop(self):
        self.running = False
        self.t.join()
        self.cap.release()
        # Clear the queue
        while not self.q.empty():
            try:
                self.q.get_nowait()
            except queue.Empty:
                pass

@contextmanager
def frame_lifecycle():
    """Context manager to ensure proper frame cleanup"""
    try:
        yield
    finally:
        cv2.destroyAllWindows()
        gc.collect()

def detect_objects(model, frame, memory_monitor):
    """Run object detection with memory monitoring"""
    if memory_monitor.should_cleanup():
        gc.collect()
    results = model(frame)
    return results

def save_to_csv(csv_file, detections, timestamp):
    """Save detections with minimal memory footprint"""
    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        for obj in detections:
            writer.writerow([timestamp, obj["label"], f"{obj['confidence']:.3f}"])

def save_screenshot(frame, folder, timestamp):
    """Save screenshot with compression"""
    if not os.path.exists(folder):
        os.makedirs(folder)
    filename = os.path.join(folder, f"screenshot_{timestamp}.jpg")  # Using JPG instead of PNG
    cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])


def display_stream(rtsp_url, window_name, frame_rate=28, model=None, csv_file="detections.csv", memory_monitor=None):
    """Memory-optimized stream display"""
    start_memory = monitor_memory()
    print(f"Starting memory for {window_name}: {start_memory:.2f} MB")
    
    vs = VideoCapture(rtsp_url)
    fps = 0
    frame_count = 0
    
    with frame_lifecycle():
        while not shutdown_event.is_set():
            frame, success = vs.read()
            if not success:
                break

            frame_count += 1
            if frame_count % 10 == 0 and memory_monitor.should_cleanup():
                gc.collect()

            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
            results = detect_objects(model, frame, memory_monitor)

            # Process detections efficiently
            detections = []
            human_count = 0
            
            for box in results[0].boxes:
                confidence = float(box.conf[0])
                if confidence < 0.5:  # Skip low confidence detections
                    continue
                    
                label = model.names[int(box.cls[0])]
                if label == "person":
                    human_count += 1
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append({
                    "label": label,
                    "confidence": confidence,
                    "bbox": (x1, y1, x2, y2)
                })

            # Save data only when necessary
            if detections:
                save_to_csv(csv_file, detections, timestamp)

            if human_count >= 2:
                save_screenshot(frame, "screenshots", timestamp)

            # Clean up results object
            del results
            
            # Display frame with minimal overhead
            if frame_count % 2 == 0:  # Update display every other frame
                for det in detections:
                    x1, y1, x2, y2 = det["bbox"]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                cv2.imshow(window_name, frame)
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Clean up frame
            del frame
            
    end_memory = monitor_memory()
    print(f"Ending memory for {window_name}: {end_memory:.2f} MB")
    print(f"Memory change: {end_memory - start_memory:.2f} MB")
    vs.stop()

def run_streams(rtsp_urls, frame_rate=14):
    """Run streams with memory monitoring"""
    initial_memory = monitor_memory()
    print(f"Initial system memory: {initial_memory:.2f} MB")
    
    model = YOLO("yolo11n.pt")
    memory_monitor = MemoryMonitor(threshold_mb=400)
    
    threads = []
    for i, url in enumerate(rtsp_urls):
        t = threading.Thread(
            target=display_stream,
            args=(url, f"Stream {i + 1}", frame_rate, model, "detections.csv", memory_monitor)
        )
        t.daemon = True
        t.start()
        threads.append(t)

    # Monitor memory usage
    peak_memory = initial_memory
    while any(t.is_alive() for t in threads):
        current_memory = monitor_memory()
        peak_memory = max(peak_memory, current_memory)
        if memory_monitor.should_cleanup():
            gc.collect()
        time.sleep(10)

    final_memory = monitor_memory()
    print(f"\nMemory Statistics:")
    print(f"Initial: {initial_memory:.2f} MB")
    print(f"Peak: {peak_memory:.2f} MB")
    print(f"Final: {final_memory:.2f} MB")
    print(f"Overall change: {final_memory - initial_memory:.2f} MB")

    for t in threads:
        t.join()

def load_rtsp_urls():
    """
    Load RTSP URLs from the .env file.
    """
    urls = []
    index = 1
    while True:
        url = os.getenv(f"RTSP_URL_{index}")
        if not url:
            break
        urls.append(url)
        index += 1
    return urls

def signal_handler(signal, frame):
    """
    Handle graceful shutdown signals.
    """
    print("Shutting down...")
    shutdown_event.set()

def main():
    """Main function with memory management"""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    rtsp_urls = load_rtsp_urls()
    if not rtsp_urls:
        print("No RTSP URLs found in .env file.")
        return

    try:
        run_streams(rtsp_urls)
    finally:
        cv2.destroyAllWindows()
        gc.collect()

if __name__ == "__main__":
    main()