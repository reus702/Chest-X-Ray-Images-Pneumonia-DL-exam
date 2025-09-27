import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

# Cartelle
normal_dir = "Dataset - Copia/train/NORMAL"
pneumonia_dir = "Dataset - Copia/train/PNEUMONIA"

# Conta quante immagini ci sono
n_normal = len(os.listdir(normal_dir))
n_pneumonia = len(os.listdir(pneumonia_dir))
extra_needed = n_pneumonia - n_normal

print(f"NORMAL: {n_normal}, PNEUMONIA: {n_pneumonia}, da generare: {extra_needed}")

# Data Augmentation
datagen = ImageDataGenerator(
    # width_shift_range=0.05,
    # height_shift_range=0.05,
    # shear_range=0.05,
    # zoom_range=0.05,
    # # brightness_range=[0.8, 1.2],#add this
    # #horizontal_flip=True,
    # fill_mode="constant"
)
i = 0
for img_name in os.listdir(normal_dir):
    img_path = os.path.join(normal_dir, img_name)
    img = load_img(img_path)  
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)

    for batch in datagen.flow(
        x,
        batch_size=1,
        save_to_dir=normal_dir,  
        save_prefix="aug_normal",
        save_format="jpeg"
    ):
        i += 1
        if i >= extra_needed:  
            break
    if i >= extra_needed:
        break

print(f"✅ Generati {i} nuovi NORMAL augmentati in {normal_dir}")
