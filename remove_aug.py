import os

directory = "/home/diego/uni/magistrale/DL/DL-exam/Chest-X-Ray-Images-Pneumonia-DL-exam/Dataset - Copia/train/NORMAL"

for filename in os.listdir(directory):
    if filename.startswith("aug"):
        file_path = os.path.join(directory, filename)
        try:
            os.remove(file_path)
            print(f"Eliminato: {file_path}")
        except Exception as e:
            print(f"Errore eliminando {file_path}: {e}")
