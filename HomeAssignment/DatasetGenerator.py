import os
import random
import shutil
import cv2
from PIL import Image
from tqdm import tqdm

# --- CONFIGURATION ---
BACKGROUND_DIR = 'HomeAssignment/Dataset/wallpaper_dataset'
FOREGROUND_ROOT_TRAIN = 'HomeAssignment/Dataset/Foregrounds_Train'
FOREGROUND_ROOT_TEST = 'HomeAssignment/Dataset/Foregrounds_Test'
OUTPUT_BASE = 'HomeAssignment/Dataset/TestDataset060126/newest_dataset'

# TARGET CLASSES (These get labels)
TARGET_CLASSES = ['ChatGPT', 'Claude', 'Gemini']

# DISTRACTOR FOLDER (Used for negatives AND blocking target windows)
DISTRACTOR_NAME = 'distractors'

# SETTINGS
TRAIN_COPIES_PER_IMG = 100
TEST_COPIES_PER_IMG = 75
NEGATIVES_COUNT = 500

# OCCLUSION SETTINGS
OCCLUSION_PROBABILITY = 0.6  # 60% chance a window will be partially blocked
OCCLUSION_MAX_COVER = 0.9    # Max 50% of the window can be covered (so it's not totally hidden)

SCALE_MIN = 0.5
SCALE_MAX = 0.8
# ---------------------

def setup_directories():
    if os.path.exists(OUTPUT_BASE):
        shutil.rmtree(OUTPUT_BASE)
    
    for split in ['train', 'test']:
        os.makedirs(f'{OUTPUT_BASE}/images/{split}', exist_ok=True)
        os.makedirs(f'{OUTPUT_BASE}/labels/{split}', exist_ok=True)

# https://www.geeksforgeeks.org/python/python-os-listdir-method/
def get_images(folder):
    if not os.path.exists(folder): return []
    valid = ('.jpg', '.jpeg', '.png', '.bmp')
    output = []
    for f in os.listdir(folder):
        if f.lower().endswith(valid):
            output.append(os.path.join(folder, f))
    return output

def get_RGBA_image(path):
    try:
        imBGR = cv2.imread(path)
        imRGBA = cv2.cvtColor(imBGR, cv2.COLOR_BGR2RGBA)

        return imRGBA
    except:
        return None

# https://roboflow.com/formats/yolo
def convert_to_yolo(img_w, img_h, x_min, y_min, x_max, y_max):
    bw, bh = x_max - x_min, y_max - y_min
    xc, yc = x_min + (bw / 2.0), y_min + (bh / 2.0)
    return xc/img_w, yc/img_h, bw/img_w, bh/img_h

def paste_window_safe(bg, fg, scale_min, scale_max):
    """Pastes a window onto the background and returns bbox + pasted image."""
    bg_w, bg_h = bg.size
    fg_aspect = fg.width / fg.height
    
    scale = random.uniform(scale_min, scale_max)
    new_w = int(bg_w * scale)
    new_h = int(new_w / fg_aspect)

    # Height Constraint
    if new_h > bg_h * 0.95:
        new_h = int(bg_h * 0.95)
        new_w = int(new_h * fg_aspect)

    fg_resized = fg.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    max_x = max(0, bg_w - new_w)
    max_y = max(0, bg_h - new_h)
    px = random.randint(0, max_x)
    py = random.randint(0, max_y)

    # Create a copy so we don't modify the original yet
    comp = bg.copy()
    comp.paste(fg_resized, (px, py), fg_resized)
    
    # Return image, bbox coordinates, and the actual resized foreground object
    return comp, (px, py, px+new_w, py+new_h), fg_resized

def apply_occlusion(base_img, target_bbox, distractors):
    """
    Pastes a distractor ON TOP of the target window to simulate overlap.
    """
    if not distractors: return base_img
    
    # Load a random distractor
    dist_path = random.choice(distractors)
    dist_img = get_RGBA_image(dist_path)# Image.open(dist_path).convert("RGBA")
    
    # Target window coordinates
    tx1, ty1, tx2, ty2 = target_bbox
    target_w = tx2 - tx1
    target_h = ty2 - ty1
    
    # Resize distractor to be smaller than the target (so it doesn't hide it 100%)
    # We want it to be roughly 30-60% of the target's size
    scale = random.uniform(0.3, 0.6) 
    d_aspect = dist_img.width / dist_img.height
    d_new_w = int(target_w * scale)
    d_new_h = int(d_new_w / d_aspect)
    
    dist_resized = dist_img.resize((d_new_w, d_new_h), Image.Resampling.LANCZOS)
    
    # Pick a paste position that GUARANTEES overlap with the target
    # We pick a point inside the target box to anchor the distractor
    overlap_x = random.randint(tx1 - (d_new_w // 2), tx2 - (d_new_w // 2))
    overlap_y = random.randint(ty1 - (d_new_h // 2), ty2 - (d_new_h // 2))
    
    # Paste distractor on top
    base_img.paste(dist_resized, (overlap_x, overlap_y), dist_resized)
    
    return base_img

def process_partition(split_name, fg_root, bg_images, class_map, copies_per_img, negativeCount):
    print(f"\n--- Processing {split_name.upper()} ---")
    global_count = 0
    
    distractor_path = os.path.join(fg_root, DISTRACTOR_NAME)
    distractors = get_images(distractor_path)
    
    if distractors:
        print(f"  Loaded {len(distractors)} distractors for occlusion generation.")
    else:
        print("  [Warning] No distractors found! Occlusion will be skipped.")

    # Iterate Target Classes
    for class_name, class_id in class_map.items():
        folder_path = os.path.join(fg_root, class_name)
        foregrounds = get_images(folder_path)
        
        if not foregrounds: continue
        print(f"  Class '{class_name}' (ID: {class_id}): Generating...")

        for fg_path in foregrounds:
            try:
                fg_original = get_RGBA_image(fg_path) #Image.open(fg_path).convert("RGBA")
            except: continue

            for _ in range(copies_per_img):
                try:
                    bg_path = random.choice(bg_images)
                    bg = get_RGBA_image(bg_path) #Image.open(bg_path).convert("RGBA")
                    bg_w, bg_h = bg.size

                    # 1. Add 'Background Noise' (Distractor BEHIND target)
                    # This helps the model deal with messy desktops
                    if distractors and random.random() < 0.3:
                        bg, _, _ = paste_window_safe(bg, get_RGBA_image(random.choice(distractors)) , 0.4, 0.9)

                    # 2. Paste Target Window
                    final_img, (x1, y1, x2, y2), _ = paste_window_safe(bg, fg_original, SCALE_MIN, SCALE_MAX)

                    # 3. Add Occlusion (Distractor ON TOP OF target)
                    # This simulates a popup or another app blocking the view
                    if distractors and random.random() < OCCLUSION_PROBABILITY:
                        final_img = apply_occlusion(final_img, (x1, y1, x2, y2), distractors)

                    # Save Image
                    fname = f"{split_name}_{class_name}_{global_count:06d}"
                    final_img.convert("RGB").save(f"{OUTPUT_BASE}/images/{split_name}/{fname}.jpg")
                    
                    # Save Label (Note: We keep the ORIGINAL box even if occluded)
                    bbox = convert_to_yolo(bg_w, bg_h, x1, y1, x2, y2)
                    with open(f"{OUTPUT_BASE}/labels/{split_name}/{fname}.txt", 'w') as f:
                        f.write(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")

                    global_count += 1
                except Exception as e:
                    # print(e) # Uncomment to debug
                    continue

    # Generate Pure Negative Samples (Just distractors, no targets)
    if distractors:
        print(f"  Generating Distractor-Only Negatives...")
        for _ in tqdm(range(negativeCount)):
            try:
                bg = get_RGBA_image(random.choice(bg_images)) #Image.open(random.choice(bg_images)).convert("RGBA")
                dist_img = get_RGBA_image(random.choice(distractors)) #Image.open(random.choice(distractors)).convert("RGBA")
                
                final_img, _, _ = paste_window_safe(bg, dist_img, SCALE_MIN, SCALE_MAX)
                
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
    with open('homeassignment/latestdata_TEST.yaml', 'w') as f:
        f.write(yaml_content)

    print("\nGeneration Complete! Dataset includes occluded samples.")

if __name__ == "__main__":
    main()