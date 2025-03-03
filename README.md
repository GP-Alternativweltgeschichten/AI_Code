# Python Code für das KI-Modell

## 📌 Beschreibung
Dieser Code wurde entwickelt, um den Museumsbesuchern die Möglichkeit zu geben, ihre Kreativität in den verschiedenen Szenarien rund um Olpe auszuleben, und ihre eigene "Alternativweltgeschichte" zu erschaffen.
Die Codebasis umfasst Skripte zum Trainieren und Testen eines KI-Modells, zur Serververbindung mit dem Backend, zum Verarbeiten von Prompts, und ein Skript zum Testen des Servers direkt mit Python.

## 📖 Inhalt


## ⚙️ Installation
1. Installieren Sie eine Python-Entwicklungsumgebung und ein Tool zum Verwalten von Python-Umgebungen (Theoretisch optional, aber sehr hilfreich)
   In diesem Projekt wurde hauptsächlich Pycharm (https://www.jetbrains.com/de-de/pycharm/) und Anaconda (https://www.anaconda.com/download) verwendet.
3. Klonen Sie das Repository in Ihr Projektverzeichnis
4. Erstellen Sie eine Python-Umgebung mit Anaconda.
   Dazu gehen Sie in Pycharm auf *Settings* --> *Project:* "Name des Projekts" --> *Python Interpreter*.
   Wählen dann Add *Interpreter* --> *Add Local Interpreter*.
   Im neuen Fenster wählen Sie dann *Conda Environment* und *Create new environment*.
   Als *Environment name* wählen Sie einen passenden Namen (Hier: OlpeAI) und als *Python version* 3.10
5. Öffnen Sie nun in Pycharm ein Terminal.
   Vor dem Dateipfad sollte nun der Name der Anaconda-Umgebung stehen.
   Ist dies nicht der Fall, kann mit dem folgenden Befehl die Anaconda-Umgebung aktiviert werden:
   ```sh
   conda activate \<Name der Umgebung\>
   ```
7. Installieren Sie in dem Terminal alle relevanten Python-Bibliotheken.
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
/Client
  /rest_client.py
/Model_Training
  /Test_Images
    /input.png
    /mask.png
  /model_training.ipynb
/Server
  /image_processing.py
  /inpaint_REST.py
  /prompt_engineering.py
  /request_types.py
```

--- 
