import re

prompt_dictionary = {
    "river": "An aerial view of the city Olpe: a calm lake with some trees along the shore.",
    "water": "An aerial view of the city Olpe: a calm lake with some trees along the shore.",
    "church": "Satellite view of Olpe, with the Church of Olpe, a large cross-shaped building, at the center.",
    "parking": "An aerial view of the city of Olpe: a road and a building with a parking lot.",
    "houses": "An aerial view of the city of Olpe: a road with parked cars and houses.",
    "residential": "An aerial view of the city of Olpe: a road with parked cars and houses.",
    "fluss": "An aerial view of the city Olpe: a calm lake with some trees along the shore.",
    "wasser": "An aerial view of the city Olpe: a calm lake with some trees along the shore.",
    "kirche": "Satellite view of Olpe, with the Church of Olpe, a large cross-shaped building, at the center.",
    "parkplatz": "An aerial view of the city of Olpe: a road and a building with a parking lot.",
    "haus": "An aerial view of the city of Olpe: a road with parked cars and houses.",
    "häuser": "An aerial view of the city of Olpe: a road with parked cars and houses."
}

def get_enhanced_prompt(prompt : str) -> str:
    cleaned_prompt = re.sub(r'[^\w\s]', '', prompt)  # Removes punctuation
    words = cleaned_prompt.lower().split()

    enhancements = [prompt_dictionary[word] for word in words if word in prompt_dictionary]

    return ", ".join(enhancements + [prompt])
