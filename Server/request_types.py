import base64
from io import BytesIO

from PIL import Image
from pydantic import BaseModel
from deep_translator import GoogleTranslator

from image_processing import convert_mask


# Request Typen
class InpaintRequest(BaseModel):
    prompt: str
    image: str
    mask: str
    realism: int

    def get_image_as_rgb(self):
        image_data = base64.b64decode(self.image.split(",")[1])
        return Image.open(BytesIO(image_data)).convert("RGB")

    def get_mask_as_rgb(self):
        mask_data = base64.b64decode(self.mask.split(",")[1])
        mask = Image.open(BytesIO(mask_data)).convert("RGBA")
        return convert_mask(mask)

    def get_prepared_prompt(self):
        new_prompt = GoogleTranslator(source='de', target='en').translate(self.prompt)
        return f'Aerial view of {new_prompt}'


class PromptRequest(BaseModel):
    prompt: str