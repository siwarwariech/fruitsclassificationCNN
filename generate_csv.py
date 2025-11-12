import os
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def load_dataset_as_dataframe(folder_path, image_size=(128, 128)):
    data, labels = [], []

    for root, dirs, files in os.walk(folder_path):
        label = os.path.basename(root)
        for file in files:
            try:
                img_path = os.path.join(root, file)
                img = load_img(img_path, target_size=image_size)
                img_array = img_to_array(img) / 255.0
                data.append(img_array.tolist())
                labels.append(label)
            except Exception as e:
                print(f"Erreur chargement {file}: {e}")

    return pd.DataFrame({'image': data, 'label': labels})

# 📦 Appelle la fonction pour générer ton DataFrame
folder_path = "images"  # <-- change si ton dossier est ailleurs
df = load_dataset_as_dataframe(folder_path)

# 💾 Sauvegarde dans ton dossier 'data/'
os.makedirs("data", exist_ok=True)
df.to_csv("data/fruits_dataset.csv", index=False)
print("✅ CSV généré avec succès dans data/fruits_dataset.csv")
