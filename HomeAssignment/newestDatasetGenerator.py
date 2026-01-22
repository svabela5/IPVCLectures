import os
import random
import shutil
from PIL import Image
from tqdm import tqdm

# --- CONFIGURATION ---
BACKGROUND_DIR = 'HomeAssignment/Dataset/wallpaper_dataset'
FOREGROUND_ROOT_TRAIN = 'HomeAssignment/Dataset/Foregrounds_Train'
FOREGROUND_ROOT_TEST = 'HomeAssignment/Dataset/Foregrounds_Test'
OUTPUT_BASE = 'HomeAssignment/Dataset/TestDataset060126/CroppedDataset'

# TARGET CLASSES
TARGET_CLASSES = ['ChatGPT', 'Claude', 'Gemini']

# DISTRACTOR FOLDER
DISTRACTOR_NAME = 'distractors'

# SETTINGS
TRAIN_COPIES_PER_IMG = 100
TEST_COPIES_PER_IMG = 75
NEGATIVES_COUNT = 1500

# CROP SETTINGS
# When cropping, keep at least this much of the original width/height
# (e.g., 0.4 means the crop will be at least 40% of the original window size)
MIN_CROP_RATIO = 0.4 

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

def get_random_crop(img):
    """
    Crops the image to a random size, keeping at least MIN_CROP_RATIO of the original.
    """
    w, h = img.size
    
    # Determine the random crop size
    # We ensure the crop is at least MIN_CROP_RATIO (e.g. 40%) of the total size
    min_w = int(w * MIN_CROP_RATIO)
    min_h = int(h * MIN_CROP_RATIO)
    
    # If image is too small to crop safely, return original
    if min_w >= w or min_h >= h:
        return img

    crop_w = random.randint(min_w, w)
    crop_h = random.randint(min_h, h)
    
    # Determine random position for the crop
    x = random.randint(0, w - crop_w)
    y = random.randint(0, h - crop_h)
    return img
    return img.crop((x, y, x + crop_w, y + crop_h))

def paste_window_simple(bg, fg):
    """
    Pastes fg onto bg. 
    Only scales down if fg is strictly larger than bg.
    """
    bg_w, bg_h = bg.size
    fg_w, fg_h = fg.size

    # Safety: If foreground (or crop) is bigger than background, shrink it to fit
    if fg_w > bg_w or fg_h > bg_h:
        width_ratio = bg_w / fg_w
        height_ratio = bg_h / fg_h
        scale_factor = min(width_ratio, height_ratio) * 0.95
        
        new_w = int(fg_w * scale_factor)
        new_h = int(fg_h * scale_factor)
        fg_resized = fg.resize((new_w, new_h), Image.Resampling.LANCZOS)
    else:
        fg_resized = fg
        new_w, new_h = fg_resized.size
    
    # Random Position on Wallpaper
    max_x = max(0, bg_w - new_w)
    max_y = max(0, bg_h - new_h)
    px = random.randint(0, max_x)
    py = random.randint(0, max_y)

    comp = bg.copy()
    comp.paste(fg_resized, (px, py), fg_resized)
    
    return comp, (px, py, px+new_w, py+new_h)

def process_partition(split_name, fg_root, bg_images, class_map, copies_per_img, DistractorsAmt):
    print(f"\n--- Processing {split_name.upper()} ---")
    global_count = 0
    
    # 1. Load Distractors
    distractor_path = os.path.join(fg_root, DISTRACTOR_NAME)
    distractors = get_images(distractor_path)
    if distractors:
        print(f"  Loaded {len(distractors)} distractors.")

    # 2. Process Targets
    for class_name, class_id in class_map.items():
        folder_path = os.path.join(fg_root, class_name)
        foregrounds = get_images(folder_path)
        
        if not foregrounds: continue
        print(f"  Class '{class_name}' (ID: {class_id}): Generating...")

        for fg_path in foregrounds:
            try:
                fg_original = Image.open(fg_path).convert("RGBA")
            except: continue

            for i in range(copies_per_img):
                try:
                    bg_path = random.choice(bg_images)
                    bg = Image.open(bg_path).convert("RGBA")
                    bg_w, bg_h = bg.size

                    # --- LOGIC CHANGE HERE ---
                    # Iteration 0: Use the FULL original image
                    # Iteration 1+: Use a RANDOM CROP of the image
                    if i == 0:
                        fg_to_use = fg_original
                    else:
                        fg_to_use = get_random_crop(fg_original)

                    # Paste (Algorithm handles placement and safety scaling)
                    final_img, (x1, y1, x2, y2) = paste_window_simple(bg, fg_to_use)

                    # Save Image
                    fname = f"{split_name}_{class_name}_{global_count:06d}"
                    final_img.convert("RGB").save(f"{OUTPUT_BASE}/images/{split_name}/{fname}.jpg")
                    
                    # Save Label
                    bbox = convert_to_yolo(bg_w, bg_h, x1, y1, x2, y2)
                    with open(f"{OUTPUT_BASE}/labels/{split_name}/{fname}.txt", 'w') as f:
                        f.write(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")

                    global_count += 1
                except Exception as e:
                    continue

    # 3. Generate Distractor Negatives
    if distractors:
        print(f"  Generating {DistractorsAmt} Distractor-Only Negatives...")
        for _ in tqdm(range(DistractorsAmt)):
            try:
                bg = Image.open(random.choice(bg_images)).convert("RGBA")
                
                # Apply same logic to distractors? 
                # Let's crop them too so the AI learns "Partial Notepad" is also NOT Gemini.
                dist_original = Image.open(random.choice(distractors)).convert("RGBA")
                if random.random() > 0.5:
                    dist_to_use = get_random_crop(dist_original)
                else:
                    dist_to_use = dist_original
                
                final_img, _, = paste_window_simple(bg, dist_to_use)
                
                fname = f"{split_name}_neg_{global_count:06d}"
                final_img.convert("RGB").save(f"{OUTPUT_BASE}/images/{split_name}/{fname}.jpg")
                
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
    
    TARGET_CLASSES.sort()
    class_map = {name: i for i, name in enumerate(TARGET_CLASSES)}
    print(f"Class Mapping: {class_map}")

    process_partition('train', FOREGROUND_ROOT_TRAIN, train_bgs, class_map, TRAIN_COPIES_PER_IMG, NEGATIVES_COUNT)
    process_partition('test', FOREGROUND_ROOT_TEST, test_bgs, class_map, TEST_COPIES_PER_IMG, int(NEGATIVES_COUNT * 0.1))

    yaml_content = f"""
path: {os.path.abspath(OUTPUT_BASE)}
train: images/train
val: images/test
test: images/test

nc: {len(TARGET_CLASSES)}
names: {TARGET_CLASSES}
    """
    with open('CroppedDatasetdata.yaml', 'w') as f:
        f.write(yaml_content)

    print("\nGeneration Complete! 1st image is full, subsequent are random crops.")

if __name__ == "__main__":
    main()