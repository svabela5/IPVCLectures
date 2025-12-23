import os
import random
import shutil
from PIL import Image
from tqdm import tqdm

# --- CONFIGURATION ---
BACKGROUND_DIR = 'HomeAssignment/Dataset/wallpaper_dataset'   # Your downloaded wallpapers
FOREGROUND_ROOT = 'HomeAssignment/Dataset/AIS'        # Folder containing your app subfolders (gemini, etc)
OUTPUT_BASE = 'HomeAssignment/Dataset/Dataset'    

# GENERATION SETTINGS
MIN_COPIES_PER_IMAGE = 250   # Min times EACH specific screenshot is generated
MAX_COPIES_PER_IMAGE = 500   # Max times EACH specific screenshot is generated

# NEGATIVE SAMPLES (Empty Wallpapers)
# Since we are generating thousands of positives, we need a good chunk of negatives.
TOTAL_EMPTY_IMAGES = 500     

WINDOW_SCALE_MIN = 0.3
WINDOW_SCALE_MAX = 0.8
# ---------------------

def setup_directories():
    if os.path.exists(OUTPUT_BASE):
        shutil.rmtree(OUTPUT_BASE)
    
    # Create training folders
    os.makedirs(f'{OUTPUT_BASE}/images/train', exist_ok=True)
    os.makedirs(f'{OUTPUT_BASE}/labels/train', exist_ok=True)

def get_images(folder):
    valid = ('.jpg', '.jpeg', '.png', '.bmp')
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(valid)]

def convert_to_yolo(img_w, img_h, x_min, y_min, x_max, y_max):
    bw, bh = x_max - x_min, y_max - y_min
    xc, yc = x_min + (bw / 2.0), y_min + (bh / 2.0)
    return xc/img_w, yc/img_h, bw/img_w, bh/img_h

def generate_dataset():
    setup_directories()

    # 1. Load Backgrounds
    backgrounds = get_images(BACKGROUND_DIR)
    if not backgrounds:
        print("Error: No wallpapers found!")
        return

    # 2. Map Classes (Folders -> ID)
    if not os.path.exists(FOREGROUND_ROOT):
        print(f"Error: '{FOREGROUND_ROOT}' folder not found.")
        return

    subfolders = [f.name for f in os.scandir(FOREGROUND_ROOT) if f.is_dir()]
    subfolders.sort()
    class_map = {name: i for i, name in enumerate(subfolders)}
    
    print(f"Classes found: {class_map}")
    print(f"Wallpapers available: {len(backgrounds)}")
    
    global_count = 0

    # --- STEP 1: GENERATE POSITIVES (Windows) ---
    print("\n--- Generating Positive Samples ---")
    
    for class_name, class_id in class_map.items():
        folder_path = os.path.join(FOREGROUND_ROOT, class_name)
        foregrounds = get_images(folder_path)
        
        print(f"Proc bessing class '{class_name}' ({len(foregrounds)} source images)...")

        for fg_path in foregrounds:
            # Determine how many times to replicate this specific screenshot
            num_variations = random.randint(MIN_COPIES_PER_IMAGE, MAX_COPIES_PER_IMAGE)
            
            # Load the foreground once to save I/O
            try:
                fg_original = Image.open(fg_path).convert("RGBA")
            except Exception as e:
                print(f"Could not load {fg_path}: {e}")
                continue

            for i in range(num_variations):
                try:
                    # Pick a random background for every variation
                    bg_path = random.choice(backgrounds)
                    bg = Image.open(bg_path).convert("RGBA")
                    bg_w, bg_h = bg.size

                    # -- Random Resize & Position Logic --
                    scale = random.uniform(WINDOW_SCALE_MIN, WINDOW_SCALE_MAX)
                    fg_aspect = fg_original.width / fg_original.height
                    new_w = int(bg_w * scale)
                    new_h = int(new_w / fg_aspect)

                    # Constrain height
                    if new_h > bg_h * 0.95:
                        new_h = int(bg_h * 0.95)
                        new_w = int(new_h * fg_aspect)

                    fg_resized = fg_original.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    
                    max_x = max(0, bg_w - new_w)
                    max_y = max(0, bg_h - new_h)
                    px = random.randint(0, max_x)
                    py = random.randint(0, max_y)

                    # Paste
                    comp = bg.copy()
                    comp.paste(fg_resized, (px, py), fg_resized)
                    
                    # Save
                    final_name = f"pos_{global_count:06d}"
                    comp.convert("RGB").save(f"{OUTPUT_BASE}/images/train/{final_name}.jpg")

                    # Label
                    bbox = convert_to_yolo(bg_w, bg_h, px, py, px+new_w, py+new_h)
                    with open(f"{OUTPUT_BASE}/labels/train/{final_name}.txt", 'w') as f:
                        f.write(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")

                    global_count += 1
                    
                except Exception as e:
                    # If a specific background fails, just skip it
                    continue

    # --- STEP 2: GENERATE NEGATIVES (Empty Wallpapers) ---
    print(f"\n--- Generating {TOTAL_EMPTY_IMAGES} Negative Samples (Empty Backgrounds) ---")
    
    for i in tqdm(range(TOTAL_EMPTY_IMAGES)):
        try:
            bg_path = random.choice(backgrounds)
            bg = Image.open(bg_path).convert("RGB")
            
            final_name = f"neg_{i:06d}"
            bg.save(f"{OUTPUT_BASE}/images/train/{final_name}.jpg")
            
            # Create EMPTY label file
            with open(f"{OUTPUT_BASE}/labels/train/{final_name}.txt", 'w') as f:
                pass # Writing nothing implies "no objects"
                
        except Exception:
            continue

    # --- STEP 3: CREATE CONFIG ---
    yaml_content = f"""
train: {os.path.abspath(OUTPUT_BASE)}/images/train
val: {os.path.abspath(OUTPUT_BASE)}/images/train
nc: {len(subfolders)}
names: {subfolders}
    """
    with open('data.yaml', 'w') as f:
        f.write(yaml_content)

    print(f"\nDataset Generation Complete!")
    print(f"Total Images Created: {global_count + TOTAL_EMPTY_IMAGES}")
    print(f"  - Positive Samples: {global_count}")
    print(f"  - Negative Samples: {TOTAL_EMPTY_IMAGES}")
    print(f"Configuration saved to 'data.yaml'")

if __name__ == "__main__":
    generate_dataset()