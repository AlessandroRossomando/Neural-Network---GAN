import os
from PIL import Image
import numpy as np
from torchvision import transforms



def load_images_from_folder(folder_path, num_samples=3500):
    images = []
    num_loaded = 0
    for filename in os.listdir(folder_path):
        if num_loaded < num_samples:
            img_path = os.path.join(folder_path, filename)
            image = Image.open(img_path).convert("RGB")
            images.append(np.array(image))
            num_loaded += 1
        else:
            break

    return images

def reshape_images(images):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    # Applica la trasformazione a ciascuna immagine in images
    transformed_images = [transform(img) for img in images]
    return transformed_images

# Esempio di utilizzo
if __name__ == '__main__':
    folder_path = "./img_align_celeba/img_align_celeba"
    images = load_images_from_folder(folder_path)
    img_transformed = reshape_images(images)
    '''print('len')
    print(len(images))
    print('shape')
    #images.shape
    print(images[0].shape)
    print('size')
    print(images[0].size)
    print('strides')
    print(images[0].strides)
    print('flatten')
    print(images[0].flatten())
    print(60*'-'+'img_transformed')
    print('len')
    print(len(img_transformed))
    print('shape')
    # images.shape
    print(img_transformed[0].shape)
    print('size')
    print(img_transformed[0].size)
    print('strides')
    print(img_transformed[0].stride)
    print('flatten')
    print(img_transformed[0].flatten())
'''


