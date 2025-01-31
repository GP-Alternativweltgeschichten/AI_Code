import base64
from os import pipe

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from diffusers import StableDiffusionInpaintPipeline
from diffusers import StableDiffusionPipeline
from PIL import Image
import torch
from io import BytesIO

# Modell einmalig laden
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained("stabilityai/stable-diffusion-2-1",
                                                              torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                                                              safety_checker=None,
                                                              requires_safety_checker=False)
# pipe_imgtoimg = StableDiffusionPipeline.from_pretrained("SebastianEngelberth/Olpe_Model_15k", torch_dtype=torch.float16 if device == "cuda" else torch.float32)
pipe_inpaint = pipe_inpaint.to(device)


# pipe_imgtoimg = pipe_imgtoimg.to(device)

# Request Typen
class InpaintRequest(BaseModel):
    prompt: str
    image: str
    mask: str


class PromptRequest(BaseModel):
    prompt: str


# Initialisiere die API
app = FastAPI()


@app.post("/inpaint/")
async def inpaint(
        request: InpaintRequest
):
    # Lade Eingabebilder
    initial_image = Image.open(BytesIO(await request.image.read())).convert("RGB")
    mask_image = Image.open(BytesIO(await request.mask.read())).convert("RGB")

    # Inpainting durchführen
    result = pipe(prompt=request.prompt, image=initial_image, mask_image=mask_image).images[0]

    # Ergebnis zurückgeben
    output_buffer = BytesIO()
    result.save(output_buffer, format="PNG")
    output_buffer.seek(0)
    return StreamingResponse(content=output_buffer, media_type="image/png")


@app.post("/text/")
async def text(
        request: PromptRequest
):
    # Bild generieren
    # result = pipe(prompt=request.prompt).images[0]

    result = Image.open("test.png")

    # Ergebnis zurückgeben
    output_buffer = BytesIO()
    result.save(output_buffer, format="PNG")
    output_buffer.seek(0)
    return StreamingResponse(content=output_buffer, media_type="image/png")


@app.post("/inpainting/")
async def text_and_image(
        request: InpaintRequest
):
    if request.mask is None or not request.mask.strip():
        # Image to Image
        image_data = base64.b64decode(request.image.split(",")[1])
        image = Image.open(BytesIO(image_data)).convert("RGB")

        # result = pipe_imgtoimg(prompt=request.prompt, image=image).images[0]

        output_buffer = BytesIO()
        # result.save(output_buffer, format="PNG")
        output_buffer.seek(0)
    else:
        # Inpainting
        image_data = base64.b64decode(request.image.split(",")[1])
        mask_data = base64.b64decode(request.mask.split(",")[1])

        image = Image.open(BytesIO(image_data)).convert("RGB")
        mask = Image.open(BytesIO(mask_data)).convert("RGBA")

        mask_region, new_mask = crop_masked_region(image, mask)
        new_mask.save("processed_image.png")
        mask_region.save("image_mask.png")

        result = pipe_inpaint(prompt=request.prompt, image=mask_region, mask_image=new_mask, strength=0.9,
                              num_inference_steps=200).images[0]

        final_image = insert_inpainted_region(image, result, mask)

        # Ergebnis zurückgeben
        output_buffer = BytesIO()
        final_image.save(output_buffer, format="PNG")
        output_buffer.seek(0)

    return StreamingResponse(content=output_buffer, media_type="image/png")


def crop_masked_region(image, mask):
    """
    Schneidet das Bild und die Maske auf die minimal erforderliche Größe zu.
    """
    bbox = mask.getbbox()

    if bbox:
        cropped_image = image.crop(bbox)
        cropped_mask = mask.crop(bbox)

        new_mask = Image.new("RGBA", cropped_mask.size)

        # Maske konvertieren
        for x in range(cropped_mask.width):
            for y in range(cropped_mask.height):
                r, g, b, a = cropped_mask.getpixel((x, y))

                if a == 0:
                    new_mask.putpixel((x, y), (0, 0, 0, 255))
                elif r == 0 and g == 0 and b == 0:
                    new_mask.putpixel((x, y), (255, 255, 255, 255))
                else:
                    new_mask.putpixel((x, y), (r, g, b, a))

        new_mask = new_mask.convert("RGB")

        return cropped_image, new_mask
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


# Starte den Server
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
