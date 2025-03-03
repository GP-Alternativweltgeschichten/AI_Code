import base64
from io import BytesIO

from PIL import Image
from deep_translator import GoogleTranslator
from deep_translator.exceptions import TranslationNotFound
from pydantic import BaseModel

# Request Typen
class InpaintRequest(BaseModel):
    prompt: str
    image: str
    mask: str
    model: int

    def get_image_as_rgb(self):
        image_data = base64.b64decode(self.image.split(",")[1])
        return Image.open(BytesIO(image_data)).convert("RGB")

    def get_mask_as_rgb(self):
        mask_data = base64.b64decode(self.mask.split(",")[1])
        mask = Image.open(BytesIO(mask_data)).convert("RGBA")
        return mask

    def get_prepared_prompt(self):
        try:
            translation = GoogleTranslator(source="de", target="en").translate(self.prompt)
        except TranslationNotFound:
            translation = self.prompt
        print(translation)
        return f'{translation}.'


class PromptRequest(BaseModel):
    prompt: str