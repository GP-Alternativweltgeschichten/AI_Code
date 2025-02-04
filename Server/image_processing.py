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

    return new_mask

def crop_masked_region(image, mask, padding=20):
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


def insert_inpainted_region(original_image, result_image, mask, padding=20):
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