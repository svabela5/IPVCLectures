import os
import random
import shutil
from PIL import Image
from tqdm import tqdm

# --- CONFIGURATION ---
BACKGROUND_DIR = 'HomeAssignment/Dataset/wallpaper_dataset'   # Your downloaded wallpapers
FOREGROUND_ROOT = 'HomeAssignment/Dataset/AIS'        # Folder containing your app subfolders (gemini, etc)
OUTPUT_BASE = 'HomeAssignment/Dataset/LABELEDdS'                # Where the final data will go

SAMPLES_PER_WALLPAPER = 1
NEGATIVE_RATIO = 0.2       # 20% empty wallpapers
WINDOW_SCALE_MIN = 0.3
WINDOW_SCALE_MAX = 0.8
# ---------------------

def setup_directories():
    if os.path.exists(OUTPUT_BASE):
        shutil.rmtree(OUTPUT_BASE)
    
    # YOLO requires specific structure: images/train, labels/train
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

    # 1. Scan Foregrounds to create Class Mapping
    subfolders = [f.name for f in os.scandir(FOREGROUND_ROOT) if f.is_dir()]
    subfolders.sort()  # Sort to ensure consistent ID assignment (0, 1, 2...)
    
    class_map = {name: i for i, name in enumerate(subfolders)}
    foreground_library = {}

    print(f"Found {len(subfolders)} classes: {class_map}")

    # Load images for each class
    for class_name, class_id in class_map.items():
        folder_path = os.path.join(FOREGROUND_ROOT, class_name)
        imgs = get_images(folder_path)
        if imgs:
            foreground_library[class_id] = imgs
        else:
            print(f"Warning: Folder '{class_name}' is empty!")

    if not foreground_library:
        print("No foreground images found!")
        return

    # Load Backgrounds
    backgrounds = get_images(BACKGROUND_DIR)
    print(f"Generating samples using {len(backgrounds)} wallpapers...")

    count = 0
    for bg_path in tqdm(backgrounds):
        try:
            bg = Image.open(bg_path).convert("RGBA")
            bg_w, bg_h = bg.size

            for _ in range(SAMPLES_PER_WALLPAPER):
                final_name = f"syn_{count:05d}"
                img_out = f"{OUTPUT_BASE}/images/train/{final_name}.jpg"
                lbl_out = f"{OUTPUT_BASE}/labels/train/{final_name}.txt"

                # Negative Sample Logic
                if random.random() < NEGATIVE_RATIO:
                    bg.convert("RGB").save(img_out)
                    open(lbl_out, 'w').close() # Empty label file
                else:
                    # Pick a random class and image
                    class_id = random.choice(list(foreground_library.keys()))
                    fg_path = random.choice(foreground_library[class_id])
                    
                    fg = Image.open(fg_path).convert("RGBA")

                    # Resize & Paste Logic
                    scale = random.uniform(WINDOW_SCALE_MIN, WINDOW_SCALE_MAX)
                    fg_aspect = fg.width / fg.height
                    new_w = int(bg_w * scale)
                    new_h = int(new_w / fg_aspect)
                    
                    # Prevent height overflow
                    if new_h > bg_h * 0.95:
                        new_h = int(bg_h * 0.95)
                        new_w = int(new_h * fg_aspect)
                    
                    fg_resized = fg.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    
                    max_x, max_y = max(0, bg_w - new_w), max(0, bg_h - new_h)
                    px, py = random.randint(0, max_x), random.randint(0, max_y)

                    comp = bg.copy()
                    comp.paste(fg_resized, (px, py), fg_resized)
                    comp.convert("RGB").save(img_out)

                    # Save Label
                    bbox = convert_to_yolo(bg_w, bg_h, px, py, px+new_w, py+new_h)
                    with open(lbl_out, 'w') as f:
                        f.write(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")

                count += 1
        except Exception as e:
            print(f"Skipped {bg_path}: {e}")

    # --- SAVE CONFIG FOR USER ---
    yaml_content = f"""
train: {os.path.abspath(OUTPUT_BASE)}/images/train
val: {os.path.abspath(OUTPUT_BASE)}/images/train  # (In real training, split this!)

nc: {len(subfolders)}
names: {subfolders}
    """
    with open('data.yaml', 'w') as f:
        f.write(yaml_content)

    print("\nGeneration Complete!")
    print(f"Created 'data.yaml' automatically based on your folders.")
    print(f"Class Mapping: {class_map}")

if __name__ == "__main__":
    generate_dataset()