import cv2
from ultralytics import YOLO

# --- CONFIGURATION ---
MODEL_PATH = 'HomeAssignment/AI Models/AI Detector/weights/best.pt'
INPUT_VIDEO = 'HomeAssignment/test.mkv'
OUTPUT_VIDEO = 'output_result.mp4'
CONFIDENCE_THRESHOLD = 0.5  # Only show detections with >50% confidence
# ---------------------

def process_video_custom():
    # 1. Load Model
    model = YOLO(MODEL_PATH)

    # 2. Open Input Video
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print(f"Error: Could not open video {INPUT_VIDEO}")
        return

    # 3. Get Video Properties (for saving)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # 4. Initialize Video Writer
    # 'mp4v' is a standard codec for MP4
    out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    print(f"Processing {INPUT_VIDEO} (Press 'q' to exit early)...")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # 5. Run Prediction on the current frame
        # stream=True is efficient for videos as it uses a generator
        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)

        # 6. Plot results
        # results[0].plot() returns the frame with boxes drawn on it
        annotated_frame = results[0].plot()

        # 7. Write to output file
        out.write(annotated_frame)

        # 8. Display on screen (Optional)
        cv2.imshow("YOLOv8 Window Detection", annotated_frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 9. Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Finished! Output saved to: {OUTPUT_VIDEO}")

if __name__ == "__main__":
    process_video_custom()