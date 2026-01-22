import pygetwindow as gw
import pyautogui
import random
import time
import os
import win32gui
import win32con
import json


FILE_NAME = "my_data.txt"

def load_variable():
    # Check if file exists first
    if os.path.exists(FILE_NAME):
        with open(FILE_NAME, 'r') as f:
            data = json.load(f)
            print(f"Loaded value: {data}")
            return data
    else:
        # Default value if file doesn't exist yet
        print("No file found. Starting with default value.")
        return 0
    
def save_variable(var):
    with open(FILE_NAME, 'w') as f:
        json.dump(var, f)
    print("Saved successfully.")

# --- CONFIGURATION ---
WINDOWNAME = "anti"
CLASSNAME = "distractors"
SAVE_FOLDER = f"HomeAssignment/Dataset/Foregrounds_Test/{CLASSNAME}/Auto"
TOP_CROP_HEIGHT = 0#130 
TOTAL_STEPS = 5
COUNTSTART = load_variable()
# ---------------------

def run_automation():
    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)

    screen_w, screen_h = pyautogui.size()

    windows = gw.getWindowsWithTitle(WINDOWNAME)
    if not windows:
        print("Chrome window not found.")
        return
    
    target_window = windows[0]
    hwnd = target_window._hWnd  # Get the unique ID (Handle) of the window

    print(f"Targeting: {target_window.title}")

    # 2. FORCE RESTORE (Crucial Step)
    # This sends a system command to un-maximize the window.
    # Without this, the size will be locked.
    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
    
    # Bring to front
    win32gui.SetForegroundWindow(hwnd)
    time.sleep(1) 

    for i in range(TOTAL_STEPS):
        # --- Generate Random Coordinates ---
        new_w = random.randint(500, int(screen_w * 0.8))
        new_h = random.randint(500, int(screen_h * 0.8))
        
        max_x = screen_w - new_w
        max_y = screen_h - new_h
        new_x = random.randint(0, max(0, max_x))
        new_y = random.randint(0, max(0, max_y))

        try:
            # 3. FORCE MOVE & RESIZE
            # We use win32gui.MoveWindow explicitly. 
            # It takes (Handle, X, Y, Width, Height, Repaint_Boolean)
            win32gui.MoveWindow(hwnd, new_x, new_y, new_w, new_h, True)
            
            # Wait for the window to visually update
            time.sleep(1.5)

            # --- Screenshot ---
            # Re-calculate region based on the NEW forced position
            # (Sometimes pygetwindow reads old data, so we use our variables)
            region = (
                new_x, 
                new_y + TOP_CROP_HEIGHT, 
                new_w, 
                new_h - TOP_CROP_HEIGHT
            )

            if region[3] > 0:
                filename = f"{CLASSNAME}_{COUNTSTART+i+1}.png"
                save_variable(COUNTSTART + i + 1)
                filepath = os.path.join(SAVE_FOLDER, filename)
                pyautogui.screenshot(filepath, region=region)
                print(f"[{i+1}/{TOTAL_STEPS}] Moved & Resized to {new_w}x{new_h}")
            
        except Exception as e:
            print(f"Error: {e}")

    print("Done.")

if __name__ == "__main__":
    run_automation()