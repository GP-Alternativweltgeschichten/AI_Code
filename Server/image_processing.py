from PIL import Image
from os import pipe
import numpy as np
import cv2
import trimesh
import open3d as o3d
import matplotlib.pyplot as plt

mesh = trimesh.load_mesh("./olpe_topologie.stl")

def convert_mask(mask):
    """
    Konvertiert die Maske von transparent, schwarz in schwarz, weiß.
    """
    new_mask = Image.new("RGB", mask.size)
    for x in range(mask.width):
        for y in range(mask.height):
            r, g, b, a = mask.getpixel((x, y))

            if a == 0:
                new_mask.putpixel((x, y), (0, 0, 0, 255))
            else:
                new_mask.putpixel((x, y), (255, 255, 255, 255))

    return new_mask

def crop_masked_region(image, mask, padding=50):
    """
    Schneidet das Bild und die Maske auf die minimal erforderliche Größe zu, erweitert den Bereich um `padding` Pixel,
    ohne über die Bildgrenzen hinauszugehen.
    """
    bbox = mask.getbbox()

    if bbox:
        left, upper, right, lower = bbox
        width, height = image.size

        left = max(left - padding, 0)
        upper = max(upper - padding, 0)
        right = min(right + padding, width)
        lower = min(lower + padding, height)

        cropped_image = image.crop((left, upper, right, lower))
        cropped_mask = mask.crop((left, upper, right, lower))
        return cropped_image, cropped_mask
    else:
        return None, None


def insert_inpainted_region(original_image, result_image, mask, padding=50):
    """
    Setzt das Ergebnis des Modells wieder an die richtige Stelle im ursprünglichen Bild ein,
    wobei der Bereich um `padding` Pixel erweitert wird, ohne über die Bildgrenzen hinauszugehen.
    """
    bbox = mask.getbbox()

    if not bbox:
        return original_image

    left, upper, right, lower = bbox
    width, height = original_image.size

    left = max(left - padding, 0)
    upper = max(upper - padding, 0)
    right = min(right + padding, width)
    lower = min(lower + padding, height)

    cropped_result = result_image.resize((right - left, lower - upper), Image.LANCZOS)

    result = original_image.copy()
    result.paste(cropped_result, (left, upper))

    return result

def inpaint_image(prompt, image, mask, modell_pipe: pipe):
    """
    Führt das Inpainting durch.
    """
    cropped_image, cropped_mask = crop_masked_region(image, mask)

    cropped_image.save("img.png")
    cropped_mask.save("mask.png")

    result = modell_pipe(prompt=prompt, image=cropped_image, mask_image=cropped_mask, strength=0.9,
                          num_inference_steps=200).images[0]
    return insert_inpainted_region(image, result, mask)

def transform_img(img):
    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.faces)

    # Beamer position
    proj_corners_3d = np.float32([
        vertices[0],  # Punkt 1 auf STL
        vertices[10],  # Punkt 2
        vertices[100],  # Punkt 3
        vertices[500]  # Punkt 4
    ])

    image_w, image_h = 1920, 1080  # Beamer-Auflösung
    proj_corners_2d = np.float32([
        [0, 0], [image_w, 0], [image_w, image_h], [0, image_h]
    ])

    M_forward = cv2.getPerspectiveTransform(proj_corners_2d, proj_corners_3d)

    warped_image = cv2.warpPerspective(img, M_forward, (image_w, image_h))

    plt.imshow(cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB))
    plt.title("Projektionsbild für Beamer")
    plt.show()