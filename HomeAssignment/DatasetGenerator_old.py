import os
import random
import shutil
from PIL import Image
from tqdm import tqdm

# --- CONFIGURATION ---
BACKGROUND_DIR = 'HomeAssignment/Dataset/wallpaper_dataset'
FOREGROUND_ROOT_TRAIN = 'HomeAssignment/Dataset/Foregrounds_Train'
FOREGROUND_ROOT_TEST = 'HomeAssignment/Dataset/Foregrounds_Test'
OUTPUT_BASE = 'HomeAssignment/Dataset/dataset'

# TARGET CLASSES (These get labels)
# The script will assign IDs alphabetically: ChatGPT=0, Claude=1, Gemini=2
TARGET_CLASSES = ['ChatGPT', 'Claude', 'Gemini']

# DISTRACTOR FOLDER (These get pasted but NOT labeled)
DISTRACTOR_NAME = 'distractors'

# SETTINGS
TRAIN_COPIES_PER_IMG = 300  # High volume for training
TEST_COPIES_PER_IMG = 30    # Lower volume for testing
NEGATIVES_COUNT = 500       # Empty wallpapers

SCALE_MIN = 0.3
SCALE_MAX = 0.8
# ---------------------

def setup_directories():
    if os.path.exists(OUTPUT_BASE):
        shutil.rmtree(OUTPUT_BASE)
    
    for split in ['train', 'test']:
        os.makedirs(f'{OUTPUT_BASE}/images/{split}', exist_ok=True)
        os.makedirs(f'{OUTPUT_BASE}/labels/{split}', exist_ok=True)

def get_images(folder):
    if not os.path.exists(folder): return []
    valid = ('.jpg', '.jpeg', '.png', '.bmp')
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(valid)]

def convert_to_yolo(img_w, img_h, x_min, y_min, x_max, y_max):
    bw, bh = x_max - x_min, y_max - y_min
    xc, yc = x_min + (bw / 2.0), y_min + (bh / 2.0)
    return xc/img_w, yc/img_h, bw/img_w, bh/img_h

def paste_window(bg, fg, scale_min, scale_max):
    """Resizes and pastes fg onto bg. Returns the composite image and bbox."""
    bg_w, bg_h = bg.size
    fg_aspect = fg.width / fg.height
    
    scale = random.uniform(scale_min, scale_max)
    new_w = int(bg_w * scale)
    new_h = int(new_w / fg_aspect)

    # Height Constraint (don't go off screen vertically)
    if new_h > bg_h * 0.95:
        new_h = int(bg_h * 0.95)
        new_w = int(new_h * fg_aspect)

    fg_resized = fg.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    max_x = max(0, bg_w - new_w)
    max_y = max(0, bg_h - new_h)
    px = random.randint(0, max_x)
    py = random.randint(0, max_y)

    comp = bg.copy()
    comp.paste(fg_resized, (px, py), fg_resized)
    
    return comp, (px, py, px+new_w, py+new_h)

def process_partition(split_name, fg_root, bg_images, class_map, copies_per_img):
    print(f"\n--- Processing {split_name.upper()} ---")
    global_count = 0
    
    # 1. Load Distractors (if any)
    distractor_path = os.path.join(fg_root, DISTRACTOR_NAME)
    distractors = get_images(distractor_path)
    if distractors:
        print(f"  Loaded {len(distractors)} distractor images (will be ignored by labeling).")

    # 2. Process Targets
    for class_name, class_id in class_map.items():
        folder_path = os.path.join(fg_root, class_name)
        foregrounds = get_images(folder_path)
        
        if not foregrounds: 
            print(f"  [Warning] No images for {class_name} in {split_name}")
            continue
            
        print(f"  Class '{class_name}' (ID: {class_id}): Generating from {len(foregrounds)} images...")

        for fg_path in foregrounds:
            try:
                fg_original = Image.open(fg_path).convert("RGBA")
            except: continue

            for _ in range(copies_per_img):
                try:
                    bg_path = random.choice(bg_images)
                    bg = Image.open(bg_path).convert("RGBA")
                    bg_w, bg_h = bg.size

                    # OPTIONAL: 30% chance to add a "Distractor" window BEHIND the real target
                    # This teaches the model to separate overlapping windows
                    if distractors and random.random() < 0.3:
                        dist_img = Image.open(random.choice(distractors)).convert("RGBA")
                        bg, _ = paste_window(bg, dist_img, 0.4, 0.9) # Paste distractor first

                    # Paste Target Window
                    final_img, (x1, y1, x2, y2) = paste_window(bg, fg_original, SCALE_MIN, SCALE_MAX)

                    # Save
                    fname = f"{split_name}_{class_name}_{global_count:06d}"
                    final_img.convert("RGB").save(f"{OUTPUT_BASE}/images/{split_name}/{fname}.jpg")
                    
                    # Label
                    bbox = convert_to_yolo(bg_w, bg_h, x1, y1, x2, y2)
                    with open(f"{OUTPUT_BASE}/labels/{split_name}/{fname}.txt", 'w') as f:
                        f.write(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")

                    global_count += 1
                except: continue

    # 3. Process Pure Distractors (Negative Samples)
    # We create images containing ONLY distractors and label them as EMPTY.
    if distractors:
        print(f"  Generating Distractor-Only Negatives...")
        for _ in tqdm(range(NEGATIVES_COUNT)):
            try:
                bg_path = random.choice(bg_images)
                bg = Image.open(bg_path).convert("RGBA")
                dist_img = Image.open(random.choice(distractors)).convert("RGBA")
                
                final_img, _ = paste_window(bg, dist_img, SCALE_MIN, SCALE_MAX)
                
                fname = f"{split_name}_distractor_{global_count:06d}"
                final_img.convert("RGB").save(f"{OUTPUT_BASE}/images/{split_name}/{fname}.jpg")
                
                # EMPTY LABEL FILE -> "Nothing to see here"
                with open(f"{OUTPUT_BASE}/labels/{split_name}/{fname}.txt", 'w') as f: pass
                
                global_count += 1
            except: continue

    return global_count

def main():
    setup_directories()
    
    backgrounds = get_images(BACKGROUND_DIR)
    if not backgrounds: return
    random.shuffle(backgrounds)
    
    split_idx = int(len(backgrounds) * 0.9)
    train_bgs = backgrounds[:split_idx]
    test_bgs = backgrounds[split_idx:]
    
    # Sort targets to ensure consistent IDs (ChatGPT=0, Claude=1, Gemini=2)
    TARGET_CLASSES.sort()
    class_map = {name: i for i, name in enumerate(TARGET_CLASSES)}
    print(f"Class Mapping: {class_map}")

    process_partition('train', FOREGROUND_ROOT_TRAIN, train_bgs, class_map, TRAIN_COPIES_PER_IMG)
    process_partition('test', FOREGROUND_ROOT_TEST, test_bgs, class_map, TEST_COPIES_PER_IMG)

    # Save Config
    yaml_content = f"""
path: {os.path.abspath(OUTPUT_BASE)}
train: images/train
val: images/test
test: images/test

nc: {len(TARGET_CLASSES)}
names: {TARGET_CLASSES}
    """
    with open('data.yaml', 'w') as f:
        f.write(yaml_content)

    print("\nGeneration Complete! Ready to train.")

if __name__ == "__main__":
    main()