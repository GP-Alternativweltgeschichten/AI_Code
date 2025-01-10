import requests

# Die URL des REST-Servers
url = "http://127.0.0.1:8000/text/"

# Das Prompt, das an den Server gesendet werden soll
data = {"prompt": "An aerial view of the city Olpe. Vegetation."}

# Sende POST-Anfrage mit dem Prompt als JSON
response = requests.post(url, json=data)

# Überprüfe, ob die Anfrage erfolgreich war
if response.status_code == 200:
    # Das Bild als Binärdaten erhalten
    image_data = response.content

    # Speichere das Ergebnis als PNG-Datei
    with open("result.png", "wb") as f:
        f.write(image_data)
    print("Ergebnis gespeichert als result.png")
else:
    print(f"Fehler: {response.status_code} - {response.text}")


