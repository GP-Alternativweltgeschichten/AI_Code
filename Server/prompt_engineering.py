import re

prompt_dictionary = {
    "fluss": "River.",
    "wasser": "Water. Lake.",
    "kirche": "Church.",
    "parkplatz": "Parking lot.",
    "haus": "House.",
    "häuser": "Houses.",
    "feuer": "Fire.",
    "spielplatz": "Playground.",
    "feld": "Field"
}
# Mit dem neusten Modell ist diese Funktion redundant, da einzelne Wörter gut funktionier.
# Wir lassen diese Funktion aber trotzdem drin, da so wichtige Wörter trotzdem übersetzt werden, auch wenn die übersetzungs-Funktion fehlschlägt.
# Wenn detailliertere Prompt wieder nötig werden, können diese im prompt_dictionary wieder eingesetzt werden.
def get_enhanced_prompt(prompt : str) -> str:
    cleaned_prompt = re.sub(r'[^\w\s]', '', prompt)  # Removes punctuation
    words = cleaned_prompt.lower().split()

    enhancements = [prompt_dictionary[word] for word in words if word in prompt_dictionary]

    return ", ".join(enhancements + [prompt])
