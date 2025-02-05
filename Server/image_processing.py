from PIL import Image
from os import pipe
import numpy as np
import cv2
import trimesh
import open3d as o3d


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

    transform_img(result)

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
    mesh = trimesh.load_mesh("./olpe_topologie.stl")

    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.faces)

    x_min, x_max = vertices[:, 0].min(), vertices[:, 0].max()
    y_min, y_max = vertices[:, 1].min(), vertices[:, 1].max()
    z_min, z_max = vertices[:, 2].min(), vertices[:, 2].max()

    proj_corners_3d = np.float32([
        [x_min, y_min, z_max],
        [x_max, y_min, z_max],
        [x_max, y_max, z_min],
        [x_min, y_max, z_min]
    ])

    proj_corners_3d_2d = proj_corners_3d[:, :2]
    proj_corners_3d_2d = np.float32([
        [100,100], [500,100], [500,600], [100,600]
    ])

    image_w, image_h = 1920, 1080  # Beamer-Auflösung
    proj_corners_2d = np.float32([
        [0, 0], [image_w, 0], [image_w, image_h], [0, image_h]
    ])

    M_forward = cv2.getPerspectiveTransform(proj_corners_2d, proj_corners_3d_2d)

    image = np.array(img, dtype=np.uint8)
    warped_image = cv2.warpPerspective(image, M_forward, (image_w, image_h))
    cv2.imwrite("warped_img.png", warped_image)
    print(proj_corners_2d)
    print(proj_corners_3d_2d)