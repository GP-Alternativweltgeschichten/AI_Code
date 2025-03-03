# Python Code für das KI-Modell

## 📌 Beschreibung
Dieser Code wurde entwickelt, um den Museumsbesuchern die Möglichkeit zu geben, ihre Kreativität in den verschiedenen Szenarien rund um Olpe auszuleben, und ihre eigene "Alternativweltgeschichte" zu erschaffen.<br>
Die Codebasis umfasst Skripte zum Trainieren und Testen eines KI-Modells, zur Serververbindung mit dem Backend, zum Verarbeiten von Prompts, und ein Skript zum Testen des Servers direkt mit Python.

## 📖 Inhalt


## ⚙️ Installation
1. Installieren Sie eine Python-Entwicklungsumgebung und ein Tool zum Verwalten von Python-Umgebungen (Theoretisch optional, aber sehr hilfreich). <br>
   In diesem Projekt wurde hauptsächlich Pycharm (https://www.jetbrains.com/de-de/pycharm/) und Anaconda (https://www.anaconda.com/download) verwendet.
3. Klonen Sie das Repository in Ihr Projektverzeichnis
4. Erstellen Sie eine Python-Umgebung mit Anaconda.<br>
   Dazu gehen Sie in Pycharm auf *Settings* --> *Project:* "Name des Projekts" --> *Python Interpreter*.<br>
   Wählen dann Add *Interpreter* --> *Add Local Interpreter*. <br>
   Im neuen Fenster wählen Sie dann *Conda Environment* und *Create new environment*. <br>
   Als *Environment name* wählen Sie einen passenden Namen (Hier: OlpeAI) und als *Python version* 3.10 <br>
5. Öffnen Sie nun in Pycharm ein Terminal. <br>
   Vor dem Dateipfad sollte nun der Name der Anaconda-Umgebung stehen. <br>
   Ist dies nicht der Fall, kann mit dem folgenden Befehl die Anaconda-Umgebung aktiviert werden:
   ```sh
   conda activate <Name der Umgebung>
   ```
7. Installieren Sie in dem Terminal alle relevanten Python-Bibliotheken. <br>
   Verwenden Sie dazu die Befehle:
   ```sh
   pip install fastapi uvicorn pydantic diffusers pillow torch requests
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
   ```
   Für das Trainieren von KI-Modellen sind zusätzliche Dateien nötig, diese sind in /Model_Training/model_training.ipynb zu finden

## 🚀 Anwendung ausführen
Starten Sie den KI-Server im Terminal:
```sh
python ./Server/inpaint_REST.py
```
Der Server läuft standardmäßig unter http://localhost:8000/

## 📂 Projektstruktur
```
/Client                     # Verzeichnis für den Test-Client
  /rest_client.py           # Python-Datei für den Test-Client
/Model_Training             # Verzeichnis für das Trainieren und Testen von KI-Modellen
  /Test_Images              # Enthält zwei Bilder zum Testen der Inpaint-Funktion von KI-Modellen
  /model_training.ipynb     # Pyhton-Notebook mit allen relevanten Anforderungen und Skripten zum Trainieren und Testen von KI-Modellen
/Server                     # Verzeichnis für die Server-Dateien
  /image_processing.py      # Python-Datei mit Funktionen zur Verarbeitung der Bilder und dem Ausführen des Inpaintings
  /inpaint_REST.py          # Python-Datei für den Server
  /prompt_engineering.py    # Python-Datei mit einer Funktion zum Verbessern von Prompts
  /request_types.py         # Python-Datei, die die Form des Requests festlegt und Funktionen zum Verbeiten dessen bereitstellt.
```

--- 
