# Python Code für das KI-Modell

## 📌 Beschreibung
Dieser Code wurde entwickelt, um den Museumsbesuchern die Möglichkeit zu geben, ihre Kreativität in den verschiedenen Szenarien rund um Olpe auszuleben, und ihre eigene "Alternativweltgeschichte" zu erschaffen.<br>
Die Codebasis umfasst Skripte zum Trainieren und Testen eines KI-Modells, zur Serververbindung mit dem Backend, zum Verarbeiten von Prompts, und ein Skript zum Testen des Servers direkt mit Python.

## 📖 Inhalt
- [Installation](#installation)
- [Anwendung ausführen](#anwendung-ausführen)
- [Datensatz](#datensatz)
- [KI-Modelle](#ki-modelle)
- [Projektstruktur](#projektstruktur)

## Vorgehen 🛫
Zunächst wurden Anforderungen an das Modell formuliert, die wie folgt lauten:
- kostenfrei
- Fähigkeit zur Generierung von Satellitenbildern
- Eignung für den lokalen Einsatz auf einem Computer
- Trainierbarkeit durch den Nutzer
  
Die Anforderungen resultieren aus einer gemeinsamen Diskussion über das Modell und dessen Funktionalitäten sowie aus den Requirements.
In der Folge wurde eine Suche nach im Internet verfügbaren Modellen initiiert, die den genannten Anforderungen entsprechen.
Primär wurden dazu Modelle mithilfe von [Automatic1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui) getestet. 
Dabei wurde erstmals evaluiert, inwiefern diese Modelle zur Generierung von Satellitenbildern befähigt sind.
Die Modelle selbst wurden von verschiedenen Websites wie [CivitAI](https://civitai.com/) und [HuggingFace](https://huggingface.co/) bezogen.
Modelle, die bereits in ihrer Basisvariante adäquate Satellitenbilder produzieren konnten, wurden anschließend auf ihre Fähigkeit zum Inpaint-Verfahren untersucht. 
Durch dieses Vorgehen konnte die Anzahl der Modelle auf eine überschaubare Anzahl reduziert werden.
Der entscheidende Schritt bestand darin zu untersuchen, ob diese Modelle durch weitere Training in der Lage sind, spezifische Bilder für Olpe zu erstellen.
Die Mehrzahl der Modelle scheiterte an dieser Herausforderung, sei es aufgrund von zu hohen Rechenanforderungen oder der Komplexität der Anforderungen.
Schlussendlich wurde das StableDiffusion 2.1-Modell selektiert, dessen Eigenschaften im Abschnitt //TODO dargelegt werden.

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

## 🗺️ Datensatz
Zum Training von eigenen KI-Modellen wurde ein eigener Datensatz mit 540 Bildern erstellt. <br>
Dabei wurden Bilder mithilfe von Google Maps und anderen Satelliten-Karten erzeugt und dann beschriftet. <br>
Es wurden hauptsächlich Bilder aus Olpe und dessen Umgebung verwendet, damit die Bilder der KI-Modelle starke Ähnlichkeiten zur Karte haben. <br>
Der Datensatz ist auf Huggingface zu finden: <br>
Satbilder (540 Bilder, beschriftet) - https://huggingface.co/datasets/GP-Alternativweltgeschichten/Satbilder <br>

## 🤖 KI-Modelle
Im laufe des Projekts wurden mehrere KI-Modelle mithilfe des Satbilder-Datensatzes trainiert. <br>
Das Training wurde mithilfe des Diffusers Repository (https://github.com/huggingface/diffusers) durchgeführt. <br>
Eine Anleitung zum Training ist in /Model_Training/model_training.ipynb zu finden. <br>
Alle Modelle basieren auf dem Stable Diffusion v2-1 Modell (https://huggingface.co/stabilityai/stable-diffusion-2-1). <br>
Die aktuellen Modelle sind auf Huggingface zu finden: <br>
OlpeAI (10k Schritte) - https://huggingface.co/GP-Alternativweltgeschichten/OlpeAI <br>
OlpeAI_Small (700 Schritte) - https://huggingface.co/GP-Alternativweltgeschichten/OlpeAI_Small <br>

## Verworfene Features 🚮
Die Mehrzahl der Features, die im Rahmen des Projektes konzipiert wurden, wurde in der vorliegenden Form realisiert. Dies ist auf die Tatsache zurückzuführen, dass sich die Projektgruppe bei der Entwicklung des KI-Modells nicht mit utopischen Zielen und Features befasst hat. 
Einige wenige Features wurden jedoch verworfen.
Ein Beispiel ist die Möglichkeit, die Text-to-Image-Funktion des Modells zu nutzen.
Diese wurde zunächst implementiert, dann jedoch durch die Inpaint-Funktion ersetzt. Dieser Schritt wurde unternommen, da die Notwendigkeit bzw. der Nutzen dieser Funktion für die Gäste nicht länger als gegeben angesehen wurde. Sollte eine Revision dieses Vorgehens in Betracht gezogen werden, wäre eine erneute Implementierung dieses Features eine mögliche Konsequenz.
Zudem wurde zu Projektbeginn die Idee diskutiert, die Karte in einer 3D-Ansicht zu präsentieren, in der das Modell darauf trainiert werden oder ein anderes Modell benötigt werden würde. Diese Idee wurde jedoch aus Gründen der Projektgröße verworfen.Darüber hinaus wurden weitere kleinere Features, die ähnliche Gründe wie Text-to-Image aufwiesen, entfernt.
Auch die Idee, die Stadt mit Hilfe von KI in verschiedenen Stilen wie Mittelalter oder wilder Westen zu gestalten, wurde zunächst zurückgestellt und aus Zeitgründen erstmals in den Ausblick für die zukünftige Entwicklung aufgenommen.

## Ausblick 🔭
Nach umfassender Analyse und Evaluation wurde festgestellt, dass das KI-Modell in seiner gegenwärtigen Konfiguration noch nicht die gewünschte Trainingsreife aufweist. Für die Zukunft ist demnach die weitere Optimierung des Modells auf Basis einer größeren Anzahl noch detaillierter beschrifteter Bilder vorgesehen.
Darüber hinaus ist die Verwendung eines alternativen Modells für die gleiche Aufgabe eine mögliche zukünftige Entwicklung, sollte sich dieses als überlegen erweisen.
Darüber hinaus sollte, unter Beibehaltung der Topografie, eine Integration der Topografie in das KI-Modell erwogen werden. Dafür müsste das Modell eventuell neu trainiert werden oder gar ein anderes genutzt werden. Die Integration von Flüssen, die den Berg hinabfließen oder Seen, die auf reaktive Weise auf Bergen durch das Modell generiert werden, ist eine vielversprechende Möglichkeit, die durch die Einbindung der Topografie in das KI-Modell realisiert werden könnte. 
Für die Zukunft sehen wir auch die Möglichkeit, andere Modelle als LORAS zu verwenden, um die Stadt in verschiedenen Stilen wie Mittelalter, Japanisch, Wilder Westen usw. darzustellen.
Es ist jedoch darauf hinzuweisen, dass auch ein gänzlich divergierendes Vorgehen in der Zukunft nicht ausgeschlossen werden kann. Es existieren bereits ähnliche Projekte, wie das von Nvidia ([Nvidia Canvas](https://support.nvidia.eu/hc/de/articles/360017442939-NVIDIA-CANVAS)), die demonstrieren, dass die zugrunde liegende Problemstellung nicht ausschließlich durch die vorgeschlagene Lösung gelöst werden kann. Sollte sich für ein alternatives Vorgehen entschieden werden, wäre eine entsprechende Anpassung des Modells erforderlich.

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
