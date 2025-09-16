import base64
from io import BytesIO
from PIL import Image, ImageFilter


def get_initial_prompting_text():
    dev_text = (
        "Du Bist ein chatbot, welcher darauf trainiert ist den Nutzern einer Anwendung einen geeignet prompt für die Generierung eines Bildes zu erstellen."
        "Du hilfst den Nutzern dabei, in den Kontext Passende und für das Bildgenerierungstool präzise prompts zu gestalten."
        "Du schlägst den Nutzern dann den Prompt in einem Bestimmten format vor. Bei der erstellung achtest du auf später folgende Regeln: "
        "die Chat nachrichten deinerseits sind Kurz."
        " sie sollten in etwa 2 bis Maximal 4 Sätze umfassen. Bleibe bei der Interaktion sachlich und Sympatisch. Sorge dafür, das das KontextRegelwerk im bestenfall eingehalten wird."
        "Wenn du ein Bild erhälst, ist es die Gesammte Karter mit dem Markierten Bereich welcher angepasst werden soll - Nutze dies um den Kontext zu verstehen und dem Kontext entsprechend passende Inhalte zu generieren."
        "Nutze wenn ein Prompt für die Generierung erstellt werden soll ausschließlich das Vorliegende Format ohne zusatz. Wenn ein Nutzer darauf besteht im markierten bereich etwas eher unpassendes zu generieren, kannst du ihm dabei auch helfen"
        "Regelnwerk: viele Wohnhäuser im Markierten bereich = Siedlung, viele Industrie Gebäude im Markierten bereich = Industriegebiet, viele supermärkte im markierten Berreich = Innenstad, viel Natur im Markierten Bereich = Grünanlagen/Wald/Feld, Fluss im Markierten Bereich = Flussufer" )
    system_text=("Für das senden eines Prompts nutze ausschließlich dieses Format ( füge den text an die stelle ein wo ... steht):"
                 "Zusammenfassung= ...  , Prompt= ... "
                 "Nutze dieses Format ausschließlich für Nachrichten die ein Prompt für die Bild Generierung sein sollen."
                 "Nutze für alle anderen Nachrichten keine spezielle Formatierung."
                 "Sende nur Prompt Vorschläge, wenn du den Kontext durch ein Bild erhälst.")
    return dev_text,system_text

def add_mask_outline_to_image(image, mask, color=(255, 0, 0), thickness=3):
    """
    Zeichnet den Umriss der Maske in einer bestimmten Farbe (Standard: Rot) auf das Bild.

    :param image: PIL.Image - Originalbild
    :param mask: PIL.Image (RGBA oder L-Modus) - Maske, bei der der markierte Bereich != transparent ist
    :param color: Tuple - Farbe für den Umriss (R, G, B)
    :param thickness: int - Dicke des Umrisses in Pixeln
    :return: PIL.Image - Bild mit Umriss
    """

    binary_mask = mask.convert("L").point(lambda p: 255 if p > 0 else 0, mode="1")

    # Erstelle einen Umriss: Differenz zwischen Maske und erodierter Maske
    dilated = binary_mask.filter(ImageFilter.MaxFilter(thickness * 2 + 1))
    outline = Image.new("1", mask.size)
    outline.paste(dilated, mask=binary_mask)

    # Male den Umriss auf eine Kopie des Originalbildes
    outlined_img = image.convert("RGBA").copy()

    for x in range(mask.width):
        for y in range(mask.height):
            if dilated.getpixel((x, y)) and not binary_mask.getpixel((x, y)):
                for dx in range(-thickness // 2, thickness // 2 + 1):
                    for dy in range(-thickness // 2, thickness // 2 + 1):
                        if 0 <= x + dx < mask.width and 0 <= y + dy < mask.height:
                            outlined_img.putpixel((x + dx, y + dy), color + (255,))

    return outlined_img

def get_image_as_base64 (image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")