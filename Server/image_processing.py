from os import pipe

import requests
import os
from io import BytesIO
from openai import OpenAI
from PIL import Image


def convert_mask(mask, transparency=True):
    """
    Konvertiert die Maske, sodass der Bereich, der verändert werden soll, transparent ist,
    und der Bereich, der unverändert bleibt, deckend.
    """
    new_mask = Image.new("RGBA", mask.size)

    for x in range(mask.width):
        for y in range(mask.height):
            r, g, b, a = mask.getpixel((x, y))  # Hole den RGBA-Wert des Pixels

            if a == 0:  # Wenn der Pixel transparent ist
                new_mask.putpixel((x, y), (0, 0, 0, 255))  # Schwarz (keine Veränderung)
            else:
                if transparency:
                    new_mask.putpixel((x, y), (0, 0, 0, 0))  # Transparent (für den Bereich der geändert werden soll)
                else:
                    new_mask.putpixel((x, y), (255, 255, 255, 255))  # Weiß (für den Bereich der geändert werden soll)

    new_mask.save("./img/converted_mask.png")
    return new_mask


def crop_masked_region(image, mask, padding=30, target_size=(1024, 1024), resize=True):
    """
    Schneidet das Bild und die Maske auf die minimal erforderliche Größe zu, erweitert den Bereich um `padding` Pixel,
    und skaliert es auf `target_size`, um eine standardisierte Eingabegröße für das Modell zu gewährleisten.
    """

    # Identifiziere die Bounding Box basierend auf der Maske
    bbox = mask.getbbox()  # Holen der Bounding Box der nicht transparenten Bereiche

    if bbox:
        left, upper, right, lower = bbox
        width, height = image.size

        left = max(left - padding, 0)
        upper = max(upper - padding, 0)
        right = min(right + padding, width)
        lower = min(lower + padding, height)

        cropped_width = right - left
        cropped_height = lower - upper

        if cropped_width < cropped_height:
            diff = cropped_height - cropped_width
            expand_left = diff // 2
            expand_right = diff - expand_left

            left = max(left - expand_left, 0)
            right = min(right + expand_right, width)
        elif cropped_height < cropped_width:
            diff = cropped_width - cropped_height
            expand_upper = diff // 2
            expand_lower = diff - expand_upper

            upper = max(upper - expand_upper, 0)
            lower = min(lower + expand_lower, height)

        cropped_image = image.crop((left, upper, right, lower))
        cropped_mask = mask.crop((left, upper, right, lower))

        # Skalierung auf das Zielbildformat
        if resize:
            resized_image = cropped_image.resize(target_size, Image.LANCZOS)
            resized_mask = cropped_mask.resize(target_size, Image.LANCZOS)
            resized_image.save("./img/cropped_income_image.png")
            resized_mask.save("./img/cropped_mask.png")

            return resized_image, resized_mask, (left, upper, right, lower)

        cropped_image.save("./img/cropped_income_image.png")
        cropped_mask.save("./img/cropped_mask.png")

        return cropped_image, cropped_mask, (left, upper, right, lower)
    else:
        return None, None, None



def insert_inpainted_region(original_image, result_image, bbox):
    """
    Setzt das Ergebnis des Modells wieder an die richtige Stelle im ursprünglichen Bild ein,
    nachdem es auf die ursprüngliche Bounding Box zurückskaliert wurde.
    """
    if not bbox:
        return original_image

    left, upper, right, lower = bbox
    width, height = right - left, lower - upper

    # Skaliere das bearbeitete Ergebnis zurück auf die Originalgröße der Bounding Box
    resized_result = result_image.resize((width, height), Image.LANCZOS)

    result = original_image.copy()
    result.paste(resized_result, (left, upper))

    result_image.save("./img/cropped_result.png")
    result.save("./img/result.png")

    return result


def inpaint_image_with_custom_model(prompt, image, mask, guidance_scale, modell_pipe: pipe):
    """
    Führt das Inpainting durch.
    """
    cropped_image, cropped_mask, bbox = crop_masked_region(image, mask, resize=False)
    converted_mask = convert_mask(cropped_mask, False)

    result = modell_pipe(prompt=prompt, image=cropped_image, mask_image=converted_mask, strength=0.95, guidance_scale=guidance_scale,
                         num_inference_steps=200).images[0]
    return insert_inpainted_region(image, result, bbox)


def save_temp_image(image, filename):
    """Speichert ein PIL-Bild temporär als PNG-Datei."""
    if image.mode != "RGBA":  # Falls Bild kein Alphakanal hat, umwandeln
        image = image.convert("RGBA")
    image.save(filename, format="PNG")
    return filename


def inpaint_image_with_dalle(prompt, image, mask):
    client = OpenAI() #OPENAI_API_KEY wird als Umgebungsvariable benutzt

    cropped_image, cropped_mask, bbox = crop_masked_region(image, mask)
    converted_mask = convert_mask(cropped_mask, True)

    if not cropped_image or not converted_mask:
        print("Kein gültiger Bereich zum Bearbeiten gefunden.")
        return image  # Gibt das Originalbild zurück, falls nichts zu tun ist

    # Speichere temporäre Dateien für die API
    temp_image_path = save_temp_image(cropped_image, "temp_image.png")
    temp_mask_path = save_temp_image(converted_mask, "temp_mask.png")

    image_params = {
        "model": "dall-e-2",
        "prompt": prompt,
        "n": 1,
        "size": "1024x1024",
    }

    try:
        with open(temp_image_path, "rb") as image_file, open(temp_mask_path, "rb") as mask_file:
            images_response = client.images.edit(image=image_file, mask=mask_file, **image_params)
    except Exception as e:
        print(f"Fehler bei der API-Anfrage: {e}")
        return None
    finally:
        # Lösche temporäre Dateien nach der API-Anfrage
        os.remove(temp_image_path)
        os.remove(temp_mask_path)

    # Extrahiere das Bild aus der API-Antwort
    image_data = images_response.data[0].url  # Die API gibt eine URL zurück
    print(image_data)
    result = Image.open(BytesIO(requests.get(image_data).content))  # Lade das Bild von der URL

    return insert_inpainted_region(image, result, bbox)

