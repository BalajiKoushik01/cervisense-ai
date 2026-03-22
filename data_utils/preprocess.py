import os
import shutil
from glob import glob
from PIL import Image
import imagehash
from sklearn.model_selection import train_test_split

import uuid

RAW_DIR = "data/raw"
SSL_DIR = "data/combined/ssl_unlabelled"
SUP_DIR = "data/combined/supervised"

CLASSES = ['Normal', 'CIN1', 'CIN2', 'CIN3', 'Cancer']

def is_valid_image(path):
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            if img.size[0] < 64 or img.size[1] < 64:
                return False
        return True
    except Exception:
        return False

def get_image_hash(path):
    try:
        with Image.open(path) as img:
            return imagehash.phash(img)
    except Exception:
        return None

def setup_directories():
    os.makedirs(SSL_DIR, exist_ok=True)
    for split in ['train', 'val', 'test']:
        for cls in CLASSES:
            os.makedirs(os.path.join(SUP_DIR, split, cls), exist_ok=True)

def map_label(path):
    """Map a file path to a cervical-cancer class label based on dataset folder names."""
    lower_path = path.replace('\\', '/').lower()

    # ── SIPaKMeD ───────────────────────────────────────────────────
    # Extracted folder names: im_Superficial-Intermediate, im_Parabasal,
    #                         im_Koilocytotic, im_Metaplastic, im_Dyskeratotic
    if 'sipakmed' in lower_path:
        if 'im_superficial-intermediate' in lower_path: return 'Normal'
        if 'im_metaplastic' in lower_path:              return 'Normal'
        if 'im_parabasal' in lower_path:                return 'CIN1'
        if 'im_koilocytotic' in lower_path:             return 'CIN2'
        if 'im_dyskeratotic' in lower_path:             return 'Cancer'

    # ── CRIC Cervix ─────────────────────────────────────────────────
    # The CRIC PNG files don't carry label info in their names;
    # labels live in the CSV.  Mark as unlabelled for SSL for now.
    elif 'cric' in lower_path:
        if 'nilm' in lower_path:                        return 'Normal'
        if 'asc-us' in lower_path or 'lsil' in lower_path: return 'CIN1'
        if 'hsil' in lower_path:                        return 'CIN3'
        if 'scc' in lower_path:                         return 'Cancer'

    # ── AnnoCerv (colposcopy) ───────────────────────────────────────
    elif 'annocerv' in lower_path:
        if 'healthy' in lower_path or 'normal' in lower_path: return 'Normal'
        if 'low' in lower_path or 'cin1' in lower_path:       return 'CIN1'
        if 'cin2' in lower_path:                               return 'CIN2'
        if 'cin3' in lower_path or 'high' in lower_path:      return 'CIN3'
        if 'cancer' in lower_path:                             return 'Cancer'

    # ── Intel MobileODT ─────────────────────────────────────────────
    elif 'mobileodt' in lower_path:
        # Competition doesn't use CIN labels – treat all 3 types as
        # unlabelled for SSL; skip supervised split.
        return None

    return None

def copy_split(paths, labels, split_name, class_counts):
    for p, l in zip(paths, labels):
        ext = os.path.splitext(p)[1]
        dest = os.path.join(SUP_DIR, split_name, l, f"{uuid.uuid4()}{ext}")
        shutil.copy2(p, dest)
        class_counts[l] = class_counts.get(l, 0) + 1

def process_images(image_paths):
    seen_hashes = set()
    label_map = []
    ssl_count = 0
    for path in image_paths:
        if not is_valid_image(path):
            continue
        h = get_image_hash(path)
        if h is None or h in seen_hashes:
            continue
        seen_hashes.add(h)
        ext = os.path.splitext(path)[1]
        shutil.copy2(path, os.path.join(SSL_DIR, f"{uuid.uuid4()}{ext}"))
        ssl_count += 1
        cls = map_label(path)
        if cls:
            label_map.append((path, cls))
    return label_map, ssl_count

def main():
    setup_directories()

    print("Scanning raw datasets...")
    image_paths = (
        glob(f"{RAW_DIR}/**/*.jpg",  recursive=True)
        + glob(f"{RAW_DIR}/**/*.png",  recursive=True)
        + glob(f"{RAW_DIR}/**/*.jpeg", recursive=True)
        + glob(f"{RAW_DIR}/**/*.bmp",  recursive=True)
    )

    label_map, ssl_count = process_images(image_paths)

    # Always initialise so the final print never raises UnboundLocalError
    class_counts = {}

    if label_map:
        x_data = [item[0] for item in label_map]
        y_data = [item[1] for item in label_map]

        x_train, x_temp, y_train, y_temp = train_test_split(
            x_data, y_data, test_size=0.3, stratify=y_data, random_state=42
        )
        x_val, x_test, y_val, y_test = train_test_split(
            x_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
        )

        copy_split(x_train, y_train, 'train', class_counts)
        copy_split(x_val,   y_val,   'val',   class_counts)
        copy_split(x_test,  y_test,  'test',  class_counts)
    else:
        print("  [WARN] No labelled images found for supervised split.")
        print("         All images have been copied to the SSL pool.")

    print(f"\nTotal SSL Unlabelled Images : {ssl_count}")
    print(f"Total Labelled (supervised) : {sum(class_counts.values())}")
    if class_counts:
        print("Class distribution:")
        for cls in ['Normal', 'CIN1', 'CIN2', 'CIN3', 'Cancer']:
            cnt = class_counts.get(cls, 0)
            print(f"  {cls:<8}: {cnt}")
    
if __name__ == '__main__':
    main()
