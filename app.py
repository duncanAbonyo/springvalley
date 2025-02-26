#detector.py
import cv2
import time
import threading
import queue
import os
import csv
from dotenv import load_dotenv
import signal
from ultralytics import YOLO

# Load environment variables
load_dotenv()

# Graceful shutdown flag
shutdown_event = threading.Event()

# Bufferless VideoCapture class
class VideoCapture:
    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.q = queue.Queue()
        self.running = True
        self.t = threading.Thread(target=self._reader)
        self.t.daemon = True
        self.t.start()

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # Discard previous frame
                except queue.Empty:
                    pass
            self.q.put(frame)
            self.state = ret

    def read(self):
        return self.q.get(), self.state

    def stop(self):
        self.running = False
        self.t.join()
        self.cap.release()

def detect_objects(model, frame):
    """
    Run object detection on the frame using YOLOv8.
    """
    results = model(frame)
    return results


def save_to_csv(csv_file, detections, timestamp):
    """
    Save detected objects and their timestamps to a CSV file.
    """
    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        for obj in detections:
            writer.writerow([timestamp, obj["label"], obj["confidence"]])


def display_stream(rtsp_url, window_name, frame_rate=28, model=None, csv_file="detections.csv"):
    """
    Display an RTSP stream in a window with object detection and save results to a CSV file.
    """
    vs = VideoCapture(rtsp_url)
    fps = 0
    start_time2 = time.time()

    # Open CSV file and write header if it doesn't exist
    if not os.path.exists(csv_file):
        with open(csv_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", "Object", "Confidence"])

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    while not shutdown_event.is_set():
        frame, success = vs.read()
        if not success:
            break

        start_time = time.time()
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))

        # Run object detection
        results = detect_objects(model, frame)

        # Parse detections
        detections = []
        for box in results[0].boxes:  # YOLOv8 uses `results[0].boxes` for bounding boxes
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            confidence = box.conf[0].item()  # Confidence score
            label = model.names[int(box.cls[0].item())]  # Class label
            detections.append({"label": label, "confidence": confidence, "bbox": (x1, y1, x2, y2)})

        # Save detections to CSV
        save_to_csv(csv_file, detections, timestamp)

        # Annotate frame with detections
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = det["label"]
            confidence = det["confidence"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{label} ({confidence:.2f})",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        # Calculate FPS
        loop_time2 = time.time() - start_time
        if loop_time2 > 0:
            fps = 0.9 * fps + 0.1 / loop_time2
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow(window_name, frame)
        delay = max(1, int((1 / frame_rate - loop_time2) * 1000))
        key = cv2.waitKey(delay) & 0xFF

        if key == ord("q") or shutdown_event.is_set():
            break

    total_time = time.time() - start_time2
    print(f"Total time for {window_name}: {total_time:.2f} seconds")

    cv2.destroyWindow(window_name)
    vs.stop()


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


def run_streams(rtsp_urls, frame_rate=14):
    """
    Start threads for each RTSP stream.
    """
    # Load YOLOv8 model
    # model = YOLO("best3.pt")  
    model =YOLO("best.pt")
    csv_file = "detections.csv"

    threads = []
    for i, url in enumerate(rtsp_urls):
        t = threading.Thread(target=display_stream, args=(url, f"Stream {i + 1}", frame_rate, model, csv_file))
        t.start()
        threads.append(t)

    # Wait for all threads to finish
    for t in threads:
        t.join()


def main():
    """
    Main function to load URLs and start the streaming.
    """
    signal.signal(signal.SIGINT, signal_handler)  # Handle Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Handle termination signal

    rtsp_urls = load_rtsp_urls()
    if not rtsp_urls:
        print("No RTSP URLs found in .env file.")
        return

    print("Starting streams...")
    run_streams(rtsp_urls)


if __name__ == "__main__":
    main()
