from io import BytesIO

import torch
from diffusers import StableDiffusionInpaintPipeline
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from image_processing import inpaint_image
from request_types import InpaintRequest

# Modell einmalig laden
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained("SebastianEngelberth/Olpe_Model",
                                                              torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                                                              safety_checker=None,
                                                              requires_safety_checker=False)
# pipe_imgtoimg = StableDiffusionPipeline.from_pretrained("SebastianEngelberth/Olpe_Model_15k", torch_dtype=torch.float16 if device == "cuda" else torch.float32)

pipe_inpaint = pipe_inpaint.to(device)
# pipe_imgtoimg = pipe_imgtoimg.to(device)


# Initialisiere die API
app = FastAPI()


@app.post("/inpainting/")
async def inpaint(
        request: InpaintRequest
):
    if request.mask is None or not request.mask.strip():
        # Image to Image
        image = request.get_image_as_rgb()
        # result = pipe_imgtoimg(prompt=request.prompt, image=image).images[0]

        return send_image_as_png(image)
    else:
        # Inpainting
        image = request.get_image_as_rgb()
        mask = request.get_mask_as_rgb()
        prompt = request.get_prepared_prompt()

        result = inpaint_image(prompt, image, mask, pipe_inpaint)

        # Ergebnis zurückgeben
        return send_image_as_png(result)


def send_image_as_png(image):
    output_buffer = BytesIO()
    image.save(output_buffer, format="PNG")
    output_buffer.seek(0)
    return StreamingResponse(content=output_buffer, media_type="image/png")


# Starte den Server
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
