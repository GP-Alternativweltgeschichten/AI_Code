from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# from diffusers import StableDiffusionInpaintPipeline
from diffusers import StableDiffusionPipeline
from PIL import Image
import torch
from io import BytesIO

# Modell einmalig laden
device = "cuda" if torch.cuda.is_available() else "cpu"
# pipe = StableDiffusionInpaintPipeline.from_pretrained("satbilder-test5", torch_dtype=torch.float16 if device == "cuda" else torch.float32)
pipe = StableDiffusionPipeline.from_pretrained("satbilder-test5", torch_dtype=torch.float16 if device == "cuda" else torch.float32)
pipe = pipe.to(device)

# Request Typen
class InpaintRequest(BaseModel):
    prompt: str
    image: UploadFile = File(...)
    mask : UploadFile = File(...)

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
    result = pipe(prompt=request.prompt).images[0]

    # Ergebnis zurückgeben
    output_buffer = BytesIO()
    result.save(output_buffer, format="PNG")
    output_buffer.seek(0)
    return StreamingResponse(content=output_buffer, media_type="image/png")


# Starte den Server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
