import os
import shutil

source_dir = "data/fruites"
target_dir = "images"

os.makedirs(target_dir, exist_ok=True)

for class_name in os.listdir(source_dir):
    src_class_path = os.path.join(source_dir, class_name)
    dst_class_path = os.path.join(target_dir, class_name)

    if os.path.isdir(src_class_path):
        os.makedirs(dst_class_path, exist_ok=True)
        images = os.listdir(src_class_path)[:5]  # prendre 5 images
        for img in images:
            src_img = os.path.join(src_class_path, img)
            dst_img = os.path.join(dst_class_path, img)
            shutil.copy(src_img, dst_img)

print("✅ Copie terminée. Images placées dans dossier 'images/'")
