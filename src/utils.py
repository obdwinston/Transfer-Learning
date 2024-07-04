import os
import random
import shutil
from PIL import Image

pokemon_folder = 'PokemonData'
train_folder = 'data/train'
test_folder = 'data/test'

os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

def is_image_file(filename):
    try:
        with Image.open(filename) as img:
            img.verify()
            if img.format.lower() == 'gif': # ignore .gif files
                return False
            elif img.mode == 'P' and 'transparency' in img.info:
                img = img.convert('RGBA') # convert transparent background
                img.save(filename)
            return True
    except:
        return False

class_list = []

for pokemon_name in sorted(os.listdir(pokemon_folder)):
    pokemon_path = os.path.join(pokemon_folder, pokemon_name)
    
    if os.path.isdir(pokemon_path):
        class_list.append(pokemon_name.capitalize())
        
        train_pokemon_path = os.path.join(train_folder, pokemon_name)
        test_pokemon_path = os.path.join(test_folder, pokemon_name)
        os.makedirs(train_pokemon_path, exist_ok=True)
        os.makedirs(test_pokemon_path, exist_ok=True)
        
        files = os.listdir(pokemon_path)
        
        image_files = [f for f in files if is_image_file(os.path.join(pokemon_path, f))]
        
        if len(image_files) > 5:
            test_files = random.sample(image_files, 5)
            train_files = [f for f in image_files if f not in test_files]
        else:
            test_files = image_files
            train_files = []
        
        for filename in test_files:
            shutil.copy2(os.path.join(pokemon_path, filename), os.path.join(test_pokemon_path, filename))
        
        for filename in train_files:
            shutil.copy2(os.path.join(pokemon_path, filename), os.path.join(train_pokemon_path, filename))

print("Class list:", class_list)
print("Dataset processing complete.")
