import cv2
import numpy as np
import mss
from ultralytics import YOLO

# --- CONFIGURATION ---
MODEL_PATH = 'HomeAssignment/AI Models/AI Detector/weights/best.pt'
CONFIDENCE_THRESHOLD = 0.5

# Display Settings
PREVIEW_SCALE = 0.5  # 0.5 = 50% size. Adjust this to make the window smaller/larger
MONITOR_INDEX = 1    # 1 is usually the primary monitor. Use 2 for secondary.
# ---------------------

def process_screen_capture():
    # 1. Load Model
    model = YOLO(MODEL_PATH)

    # 2. Initialize Screen Capture
    sct = mss.mss()
    
    # Check if monitor exists
    if len(sct.monitors) <= MONITOR_INDEX:
        print(f"Error: Monitor {MONITOR_INDEX} not found.")
        return

    # Get monitor dimensions
    monitor = sct.monitors[MONITOR_INDEX]
    
    # Define capture area (capturing the whole monitor here)
    capture_area = {
        "top": monitor["top"],
        "left": monitor["left"],
        "width": monitor["width"],
        "height": monitor["height"],
        "mon": MONITOR_INDEX,
    }

    print(f"Capturing Monitor {MONITOR_INDEX} ({monitor['width']}x{monitor['height']})")
    print("Press 'q' to exit.")

    while True:
        # 3. Capture the Screen
        # sct.grab returns a raw BGRA image
        sct_img = sct.grab(capture_area)
        
        # Convert to numpy array (OpenCV format) and drop the Alpha channel (BGRA -> BGR)
        frame = np.array(sct_img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # 4. Run Prediction
        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)

        # 5. Annotate Frame
        annotated_frame = results[0].plot()

        # 6. Resize for Display
        # Calculate new dimensions based on PREVIEW_SCALE
        new_width = int(annotated_frame.shape[1] * PREVIEW_SCALE)
        new_height = int(annotated_frame.shape[0] * PREVIEW_SCALE)
        
        # Resize just the view, not the detection frame
        display_frame = cv2.resize(annotated_frame, (new_width, new_height))
        
        # Show the smaller window
        cv2.imshow("YOLOv8 Screen Detection", display_frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 7. Cleanup
    cv2.destroyAllWindows()
    print("Finished.")

if __name__ == "__main__":
    process_screen_capture()