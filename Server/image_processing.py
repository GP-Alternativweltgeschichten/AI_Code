from PIL import Image
from os import pipe

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
            elif r == 0 and g == 0 and b == 0:
                new_mask.putpixel((x, y), (255, 255, 255, 255))
            else:
                new_mask.putpixel((x, y), (r, g, b, a))

    return mask

def crop_masked_region(image, mask):
    """
    Schneidet das Bild und die Maske auf die minimal erforderliche Größe zu.
    """
    bbox = mask.getbbox()

    if bbox:
        cropped_image = image.crop(bbox)
        cropped_mask = mask.crop(bbox)
        return cropped_image, cropped_mask
    else:
        return None, None


def insert_inpainted_region(original_image, result_image, mask):
    """
    Setzt das Ergebnis des Modells wieder an die richtige Stelle im ursprünglichen Bild ein.
    """
    bbox = mask.getbbox()

    if not bbox:
        return original_image

    cropped_result = result_image.resize((bbox[2] - bbox[0], bbox[3] - bbox[1]), Image.LANCZOS)
    result = original_image.copy()
    result.paste(cropped_result, (bbox[0], bbox[1]))

    return result

def inpaint_image(prompt, image, mask, modell_pipe: pipe):
    """
    Führt das Inpainting durch.
    """
    cropped_image, cropped_mask = crop_masked_region(image, mask)

    result = modell_pipe(prompt=prompt, image=cropped_image, mask_image=cropped_mask, strength=0.9,
                          num_inference_steps=200).images[0]
    return insert_inpainted_region(image, result, mask)