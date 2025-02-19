from io import BytesIO

import torch
from diffusers import StableDiffusionInpaintPipeline
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from image_processing import inpaint_image_with_custom_model, inpaint_image_with_dalle
from prompt_engineering import get_enhanced_prompt
from request_types import InpaintRequest

# Modell einmalig laden
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained("GP-Alternativweltgeschichten/Olpe_Model_05_02",
                                                              torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                                                              safety_checker=None,
                                                              requires_safety_checker=False)

pipe_inpaint = pipe_inpaint.to(device)

# Initialisiere die API
app = FastAPI()


@app.post("/inpainting/")
async def inpaint(
        request: InpaintRequest
):
    if request.mask is None or not request.mask.strip():
        # Image to Image
        image = request.get_image_as_rgb()

        return send_image_as_png(image)
    else:
        # Inpainting
        image = request.get_image_as_rgb()
        mask = request.get_mask_as_rgb()
        prompt = request.prompt
        model = request.realism

        if model != model:
            prompt = request.get_prepared_prompt()
            # Prompt Enhancing
            prompt = get_enhanced_prompt(prompt)
            result = inpaint_image_with_custom_model(prompt, image, mask, pipe_inpaint)
        else:
            result = inpaint_image_with_dalle(prompt, image, mask)

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
