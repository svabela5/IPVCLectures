import os
import random
import shutil
from PIL import Image
from tqdm import tqdm

# --- CONFIGURATION ---
# INPUT: Where your "Test" source images are
BG_SOURCE_DIR = 'HomeAssignment/Dataset/Test Materials/Backgrounds'
FG_SOURCE_ROOT = 'HomeAssignment/Dataset/Test Materials/Foregrounds'

# OUTPUT: Where the final YOLO test data goes
OUTPUT_BASE = 'HomeAssignment/Dataset/Dataset'  # We will add to the existing 'dataset' folder

# SETTINGS
COPIES_PER_IMAGE = 30       # Fewer copies for testing (e.g., 30 per screenshot)
TOTAL_EMPTY_IMAGES = 50     # Test detecting "nothing" (Negative samples)
WINDOW_SCALE_MIN = 0.3
WINDOW_SCALE_MAX = 0.8
# ---------------------

def setup_directories():
    # We do NOT remove the whole dataset folder, we only reset the test subfolders
    test_img_dir = f'{OUTPUT_BASE}/images/test'
    test_lbl_dir = f'{OUTPUT_BASE}/labels/test'

    if os.path.exists(test_img_dir): shutil.rmtree(test_img_dir)
    if os.path.exists(test_lbl_dir): shutil.rmtree(test_lbl_dir)
    
    os.makedirs(test_img_dir, exist_ok=True)
    os.makedirs(test_lbl_dir, exist_ok=True)

def get_images(folder):
    if not os.path.exists(folder): return []
    valid = ('.jpg', '.jpeg', '.png', '.bmp')
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(valid)]

def convert_to_yolo(img_w, img_h, x_min, y_min, x_max, y_max):
    bw, bh = x_max - x_min, y_max - y_min
    xc, yc = x_min + (bw / 2.0), y_min + (bh / 2.0)
    return xc/img_w, yc/img_h, bw/img_w, bh/img_h

def generate_test_set():
    setup_directories()

    # 1. Validation Checks
    backgrounds = get_images(BG_SOURCE_DIR)
    if not backgrounds:
        print(f"CRITICAL ERROR: No background images found in {BG_SOURCE_DIR}")
        print("Please create this folder and add wallpapers the AI has NEVER seen.")
        return

    if not os.path.exists(FG_SOURCE_ROOT):
        print(f"CRITICAL ERROR: {FG_SOURCE_ROOT} does not exist.")
        return

    # 2. Map Classes (Must match Training IDs!)
    # We look at the folders in 'foregrounds' to ensure class IDs (0, 1, 2) stay consistent
    subfolders = [f.name for f in os.scandir(FG_SOURCE_ROOT) if f.is_dir()]
    subfolders.sort()
    class_map = {name: i for i, name in enumerate(subfolders)}
    
    print(f"Generating Test Data for classes: {class_map}")
    
    global_count = 0

    # --- PART 1: POSITIVE SAMPLES (Windows) ---
    for class_name, class_id in class_map.items():
        folder_path = os.path.join(FG_SOURCE_ROOT, class_name)
        foregrounds = get_images(folder_path)
        
        for fg_path in foregrounds:
            # Load FG once
            try:
                fg_original = Image.open(fg_path).convert("RGBA")
            except: continue

            for i in range(COPIES_PER_IMAGE):
                try:
                    bg_path = random.choice(backgrounds)
                    bg = Image.open(bg_path).convert("RGBA")
                    bg_w, bg_h = bg.size

                    # Resize & Position
                    scale = random.uniform(WINDOW_SCALE_MIN, WINDOW_SCALE_MAX)
                    fg_aspect = fg_original.width / fg_original.height
                    new_w = int(bg_w * scale)
                    new_h = int(new_w / fg_aspect)

                    if new_h > bg_h * 0.95:
                        new_h = int(bg_h * 0.95)
                        new_w = int(new_h * fg_aspect)

                    fg_resized = fg_original.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    
                    max_x, max_y = max(0, bg_w - new_w), max(0, bg_h - new_h)
                    px, py = random.randint(0, max_x), random.randint(0, max_y)

                    comp = bg.copy()
                    comp.paste(fg_resized, (px, py), fg_resized)
                    
                    final_name = f"test_pos_{global_count:05d}"
                    comp.convert("RGB").save(f"{OUTPUT_BASE}/images/test/{final_name}.jpg")

                    bbox = convert_to_yolo(bg_w, bg_h, px, py, px+new_w, py+new_h)
                    with open(f"{OUTPUT_BASE}/labels/test/{final_name}.txt", 'w') as f:
                        f.write(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")

                    global_count += 1
                except: continue

    # --- PART 2: NEGATIVE SAMPLES (Empty) ---
    print("Generating Negative Test Samples...")
    for i in range(TOTAL_EMPTY_IMAGES):
        try:
            bg_path = random.choice(backgrounds)
            bg = Image.open(bg_path).convert("RGB")
            final_name = f"test_neg_{i:05d}"
            
            bg.save(f"{OUTPUT_BASE}/images/test/{final_name}.jpg")
            # Empty label file
            with open(f"{OUTPUT_BASE}/labels/test/{final_name}.txt", 'w') as f: pass
        except: continue

    print(f"\nTest Set Generated in '{OUTPUT_BASE}/images/test'")
    print(f"Total Test Images: {global_count + TOTAL_EMPTY_IMAGES}")
    
    # Update data.yaml to include the test path
    update_yaml_config()

def update_yaml_config():
    """Updates data.yaml to point to the new test set."""
    yaml_path = 'data.yaml'
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as f:
            lines = f.readlines()
        
        # Check if 'test:' is already defined
        has_test = any(line.strip().startswith('test:') for line in lines)
        
        if not has_test:
            with open(yaml_path, 'a') as f:
                f.write(f"\ntest: {os.path.abspath(OUTPUT_BASE)}/images/test\n")
            print("Updated 'data.yaml' with test dataset path.")

if __name__ == "__main__":
    generate_test_set()