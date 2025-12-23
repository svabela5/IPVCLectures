import os
import requests
import time

# --- CONFIGURATION ---
# Add as many access keys as you have here
ACCESS_KEYS = [
    '9fUuFDxySKusml535DgzcEW9DytqgRXyhhqQQZaaFsA',
    'DYYKCFR3PrXi_4Zg-NagffkU0NSPgQp2W84qVnNOZ70',
    'zjno8nnY9j97lEDZd47Yvba7iI41uLzLr7EAIqL6t6w',
    '5laxToS_R2_8rsaceek_NcZC2K_0Wp-E1SwIzqYzp0w'
]
SAVE_FOLDER = 'HomeAssignment/Dataset/Test Materials/Backgrounds'
PROGRESS_FILE = 'HomeAssignment/progress.txt'
TOTAL_IMAGES = 1050
KEYWORDS = ['nature', 'abstract', 'minimalist', 'space', 'architecture']
# ---------------------

class KeyManager:
    """Handles rotating through multiple API keys when limits are hit."""
    def __init__(self, keys):
        self.keys = keys
        self.current_index = 0
        self.cooldowns = {key: 0 for key in keys}

    def get_key(self):
        start_index = self.current_index
        while True:
            key = self.keys[self.current_index]
            # Check if key is out of cooldown (1 hour safety)
            if time.time() > self.cooldowns[key]:
                return key
            
            # Move to next key
            self.current_index = (self.current_index + 1) % len(self.keys)
            
            # If we've checked all keys and all are in cooldown
            if self.current_index == start_index:
                wait_time = min(self.cooldowns.values()) - time.time()
                if wait_time > 0:
                    print(f"All keys limited. Waiting {int(wait_time/60)} mins...")
                    time.sleep(wait_time + 5)
        
    def mark_limited(self, key):
        print(f"Key {key[:8]}... hit rate limit. Rotating...")
        self.cooldowns[key] = time.time() + 3600 # 1 hour cooldown
        self.current_index = (self.current_index + 1) % len(self.keys)

def get_last_state():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            try: return int(f.read().strip())
            except: return 0
    return 0

def save_state(count):
    with open(PROGRESS_FILE, 'w') as f:
        f.write(str(count))

def download_wallpapers():
    if not os.path.exists(SAVE_FOLDER): os.makedirs(SAVE_FOLDER)
    
    key_manager = KeyManager(ACCESS_KEYS)
    downloaded = get_last_state()
    per_page = 30
    
    print(f"Resuming with {len(ACCESS_KEYS)} keys pooled.")
    
    while downloaded < TOTAL_IMAGES:
        current_key = key_manager.get_key()
        keyword_idx = (downloaded // per_page) % len(KEYWORDS)
        current_query = KEYWORDS[keyword_idx]
        page = (downloaded // (per_page * len(KEYWORDS))) + 1

        url = "https://api.unsplash.com/search/photos"
        params = {
            'query': current_query,
            'per_page': per_page,
            'page': page,
            'orientation': 'landscape',
            'client_id': current_key
        }

        try:
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code in [403, 429]:
                key_manager.mark_limited(current_key)
                continue
            
            photos = response.json().get('results', [])
            if not photos:
                downloaded += per_page # Advance to next page/keyword
                continue

            for photo in photos:
                if downloaded >= TOTAL_IMAGES: break
                file_path = os.path.join(SAVE_FOLDER, f"img_{downloaded:04d}.jpg")

                if not os.path.exists(file_path):
                    img_data = requests.get(photo['urls']['regular'], timeout=20).content
                    with open(file_path, 'wb') as f:
                        f.write(img_data)
                    
                downloaded += 1
                save_state(downloaded)
                print(f"[{downloaded}/{TOTAL_IMAGES}] Downloaded: {current_query}")

        except Exception as e:
            print(f"Error: {e}. Retrying...")
            time.sleep(5)

if __name__ == "__main__":
    download_wallpapers()