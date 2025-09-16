from datetime import datetime
from io import BytesIO

import torch
from diffusers import StableDiffusionInpaintPipeline
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from openai import OpenAI
from fastapi.responses import JSONResponse
from starlette.responses import PlainTextResponse

from persistence import save_income, save_result
from image_processing import inpaint_image_with_custom_model, inpaint_image_with_dalle
from prompt_engineering import get_enhanced_prompt
from request_types import InpaintRequest, ChatMessageRequest
from chatBot_interaction_interface import get_initial_prompting_text, add_mask_outline_to_image, \
    get_image_as_base64

# Modell einmalig laden
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained("GP-Alternativweltgeschichten/OlpeAI",
                                                              torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                                                              safety_checker=None,
                                                              requires_safety_checker=False)

pipe_inpaint = pipe_inpaint.to(device)

# Initialisiere die API
app = FastAPI()
# chatbot variablen
inital_developer_prompting,inital_system_prompting  = get_initial_prompting_text()
client = OpenAI()
conversations = []
@app.post("/text")
async def chat_message(
        request: ChatMessageRequest
):
    print(request.text)
    if request.conversationId not in conversations:
        chat_conversation=client.conversations.create(
            items=[
                {"type": "message",
                 "role": "system",
                 "content": [
                     {"type": "input_text", "text": inital_system_prompting}]
                 },
                {"type": "message",
                 "role": "developer",
                 "content": [
                     {"type": "input_text", "text": inital_developer_prompting},]
                 }
            ]
        )
        conversations[request.conversationId]=chat_conversation.id
        currentConversationId=chat_conversation.id
    else:
        chat_conversation = client.conversations.retrieve(request.conversationId)

    if request.mask and request.image :
        mask = request.get_mask_as_rgb()
        image = request.get_image_as_rgb()
        marked_image = add_mask_outline_to_image(image, mask)
        formated_image = get_image_as_base64(marked_image)
        response = client.responses.create(
            conversation=currentConversationId,
            model="gpt-4.1-mini",
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text",
                     "text": "benutze diesen Markierten Bereich als (neunen) Kontext für die conversation"},
                    {"type": "input_image",
                     "image_url": f"data:image/png;base64,{formated_image}"},
                ]
            }]
        )
        print(response.output_text)
        return PlainTextResponse(response.output_text)

    if request.text and request.conversationId :
        response = client.responses.create(
            conversation=currentConversationId,
            model="gpt-4.1-mini",
                input = [{
        "role": "user",
        "content": [
            {"type": "input_text", "text": request.text},
        ]
    }]
        )
    print(response.output_text)
    return PlainTextResponse(response.output_text)

@app.post("/inpainting/")
async def inpaint(
        request: InpaintRequest
):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    if request.mask is None or not request.mask.strip():
        # Image to Image
        image = request.get_image_as_rgb()

        return send_image_as_png(image)
    else:
        # Inpainting
        image = request.get_image_as_rgb()
        mask = request.get_mask_as_rgb()
        prompt = request.prompt
        model = request.model
        guidance_scale = request.guidance_scale

        if model == 0:
            prompt = request.get_prepared_prompt()
            # Prompt Enhancing
            prompt = get_enhanced_prompt(prompt)
            result = inpaint_image_with_custom_model(prompt, image, mask, guidance_scale, pipe_inpaint)
        else:
            result = inpaint_image_with_dalle(prompt, image, mask)

        save_income(image,mask,prompt,model,guidance_scale,timestamp)
        save_result(result,timestamp)

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
