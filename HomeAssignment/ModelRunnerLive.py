import cv2
import numpy as np
import mss
from ultralytics import YOLO
import time

# --- CONFIGURATION ---
MODEL_PATH = 'HomeAssignment/AI Models/AI Detector 210126 Take 2/weights/best.pt'
CONFIDENCE_THRESHOLD = 0.5
OUTPUT_FILENAME = 'output_1fps.mp4'  # Name of the saved file

# Display Settings
PREVIEW_SCALE = 0.5  # 0.5 = 50% size. Adjust this to make the window smaller/larger
MONITOR_INDEX = 3    # 1 is usually the primary monitor. Use 2 for secondary.
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

    # --- NEW: INITIALIZE VIDEO WRITER ---
    # We use the full monitor resolution for saving (not the preview scale)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    # The '1' below is the playback FPS. Since we record at 1 FPS, this makes the video play at real-time speed.
    out = cv2.VideoWriter(OUTPUT_FILENAME, fourcc, 1.0, (monitor["width"], monitor["height"]))
    # ------------------------------------

    print(f"Capturing Monitor {MONITOR_INDEX} ({monitor['width']}x{monitor['height']})")
    print(f"Saving to {OUTPUT_FILENAME} at 1 FPS.")
    print("Press 'q' to exit.")

    try:
        while True:
            # Start timer for FPS control
            start_time = time.time()

            sct_img = sct.grab(capture_area)
            
            # Convert to numpy array and drop Alpha channel (BGRA -> BGR)
            frame = np.array(sct_img)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # 4. Run Prediction
            results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)

            # 5. Annotate Frame
            annotated_frame = results[0].plot()

            # --- NEW: SAVE FRAME ---
            # Write the full-resolution frame to the video file
            out.write(annotated_frame)
            # -----------------------

            # 6. Resize for Display
            new_width = int(annotated_frame.shape[1] * PREVIEW_SCALE)
            new_height = int(annotated_frame.shape[0] * PREVIEW_SCALE)
            
            display_frame = cv2.resize(annotated_frame, (new_width, new_height))
            
            cv2.imshow("YOLOv8 Screen Detection", display_frame)

            # Press 'q' to quit (Check frequently)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


            process_time = time.time() - start_time

            time_to_wait = 1.0 - process_time
            
            if time_to_wait > 0:
                time.sleep(time_to_wait)

    except KeyboardInterrupt:
        print("Interrupted by user.")

    finally:
        # 7. Cleanup
        out.release() # Save the video file properly
        cv2.destroyAllWindows()
        print(f"Finished. Video saved to {OUTPUT_FILENAME}")

if __name__ == "__main__":
    process_screen_capture()