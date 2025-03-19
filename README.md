# Python Code für das KI-Modell

## 📌 Beschreibung
Dieser Code wurde entwickelt, um den Museumsbesuchern die Möglichkeit zu geben, ihre Kreativität in den verschiedenen Szenarien rund um Olpe auszuleben, und ihre eigene "Alternativweltgeschichte" zu erschaffen.  
Die Codebasis umfasst Skripte zum Trainieren und Testen eines KI-Modells, zur Serververbindung mit dem Backend, zum Verarbeiten von Prompts, zur Bildgenerierung, zur Persistierung von Nutzereingaben, und ein Skript zum Testen des Servers in Python.

## 📖 Inhalt
- [Vorgehen](#-vorgehen)
- [Auswahl des KI-Modells](#-auswahl-des-ki-modells)
- [Datensatz](#%EF%B8%8F-datensatz)
- [Eigene KI-Modelle](#-eigene-ki-modelle)
- [Voraussetzungen](#%EF%B8%8F-voraussetzungen)
- [Server ausführen](#-server-ausführen)
- [Server Funktionalitäten](#-server-funktionalitäten)
- [KI-Prompting](#-ki-prompting)
- [KI-Modell trainieren](#-ki-modell-trainieren)
- [Ausblick](#-ausblick)
- [Verworfene Features](#-verworfene-features)
- [Projektstruktur](#-projektstruktur)


## 🛫 Vorgehen
Zunächst wurden Anforderungen an das Modell formuliert, zu finden sind diese unter [Auswahl des KI-Modells](#-auswahl-des-ki-modells).
  
Die Anforderungen resultieren aus einer gemeinsamen Diskussion über das Modell und dessen Funktionalitäten sowie aus den Requirements.
In der Folge wurde eine Suche nach im Internet verfügbaren Modellen initiiert, die den genannten Anforderungen entsprechen.
Primär wurden dazu Modelle mithilfe von [Automatic1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui) getestet. 
Dabei wurde erstmals evaluiert, inwiefern diese Modelle zur Generierung von Satellitenbildern befähigt sind.
Die Modelle selbst wurden von verschiedenen Websites wie [CivitAI](https://civitai.com/) und [HuggingFace](https://huggingface.co/) bezogen.
Modelle, die bereits in ihrer Basisvariante adäquate Satellitenbilder produzieren konnten, wurden anschließend auf ihre Fähigkeit zum Inpaint-Verfahren untersucht. 
Durch dieses Vorgehen konnte die Anzahl der Modelle auf eine überschaubare Anzahl reduziert werden.
Der entscheidende Schritt bestand nun darin zu untersuchen, ob diese Modelle durch weiteres Training in der Lage sind, spezifische Bilder für Olpe zu erstellen.
Die Mehrzahl der Modelle scheiterte an dieser Herausforderung, sei es aufgrund von zu hohen Rechenanforderungen oder der Komplexität der Anforderungen.
Schlussendlich wurde ein StableDiffusion Modell selektiert und mit der [Diffusers Bibliothek](https://github.com/huggingface/diffusers) weiter trainiert.
Die Grundlage zur Wahl des Modells ist im Abschnitt [Auswahl des KI-Modells](#-auswahl-des-ki-modells) dargelegt.
Da die Ergebnisse des selbst-trainierten Modells zunächst nicht unseren Vorstellungen entsprachen, wurde eine zusätzliche Anbindung an das [Dall-E 2 Modell](https://openai.com/index/dall-e-2/) von [OpenAI](https://openai.com/) erstellt.
Dall-E 2 ist in der Lage konsistent Satellitenbilder zu erstellen, kostet aber einen geringen Preis pro Anfrage.
Das Dall-E 2 Modell hat dabei aber nicht ein eigenes Modell ersetzt, da eine kostenlose, lokale Option vorhanden sein soll.

## 🎯 Auswahl des KI-Modells
Basierend auf einer ersten Literaturrecherche wurden folgende Herausforderungen bei der Erstellung eines KI-Modells identifiziert:
- Keine Open Source Modelle, die Satellitenbilder erstellen können 
- Keine/wenige Datensätze verfügbar, die Satellitenbilder enthalten 
- Keine/wenige Datensätze verfügbar, die dem Stadtbild von Olpe entsprechen 
- Teilweise sehr aufwendige Schritte zum Trainieren der Modelle 
- Teilweise hohe benötigte Rechenleistung
- Segmentierung der Satellitenbilder sehr aufwendig, da viele kleine Objekte vorhanden sind 
- Teilweise keine Unterstützung von InPaint, nur Text to Image 

Bezüglich der Herausforderungen musste im Laufe des Projektes ein ideales KI-Modell für unseren Anwendungsfall definiert werden.
Die folgenden Anforderungen wurden definiert:
- Open Source, kostenlos 
- Kann Satellitenbilder generieren 
- Kann lokal auf einem Rechner laufen 
- Kann von uns trainiert werden 
- Ist spezialisiert auf das Erscheinungsbild der Stadt Olpe 
- Die Erstellung eines Datensatzes und das Training sind mit angemessenem Aufwand möglich
- Unterstützt InPaint 

Nach einer Reihe von Tests und zusätzlicher Recherche wurde schließlich ein Stable Diffusion-Modell eingesetzt.
Die zuvor genannten Punkte konnten mithilfe der Diffusers-Bibliothek, die auf der [Diffusers Bibliothek](https://github.com/huggingface/diffusers) basiert, realisiert werden.
Die Diffusers Bibliothek repräsentiert den aktuellen Stand der Technik, ist Open-Source und bietet auch vortrainierte Modelle als Grundlage.
Dies erlaubt das Erstellen eines Datensatzes durch die Beschreibung von Bildern, anstatt durch aufwendiges Segmentieren.
Darüber hinaus ist das (Weiter-)Trainieren eines Modells einfach möglich.
Die Verwendung und das Weitertrainieren von vorab erstellten Modellen ermöglichte es, die Akkuratheit der Abbildung des Stadtbildes von Olpe zu gewährleisten und zudem die Umsetzung kreativerer Anfragen zu ermöglichen.
Als Grundlage wurde das [Stable Diffusion v2-1 Modell](https://huggingface.co/stabilityai/stable-diffusion-2-1) von [StabilityAI](https://stability.ai/) ausgewählt, welches eine umfangreiche Bilddatenbank enthält.
Das trainierte Modell konnte abschließend lokal mithilfe der Diffuser-Bibliothek geladen werden.
Das Modell kann in verschiedenen "Pipelines" verwendet werden, welche je nach Auswahl Text to Image, Inpaint oder weitere Aufgaben ausführen können.

## 🗺️ Datensatz
Zum Training von eigenen KI-Modellen wurde ein eigener Datensatz mit 540 Bildern erstellt.  
Dabei wurden Bilder mithilfe von Google Maps und anderen Satelliten-Karten erzeugt und dann beschriftet.
Es wurden hauptsächlich Bilder aus Olpe und dessen Umgebung verwendet, damit die Bilder der KI-Modelle starke Ähnlichkeiten zur Karte haben.   
Der Datensatz ist auf Huggingface zu finden:   
- [Satbilder](https://huggingface.co/datasets/GP-Alternativweltgeschichten/Satbilder) (540 Bilder, beschriftet)   

Die Bilder des Datensatzes befinden sich dabei in [/data/](https://huggingface.co/datasets/GP-Alternativweltgeschichten/Satbilder/tree/main/data) und die zugehörigen Beschreibungen in [/metadata.csv](https://huggingface.co/datasets/GP-Alternativweltgeschichten/Satbilder/blob/main/metadata.csv).

## 🤖 Eigene KI-Modelle
Im laufe des Projekts wurden mehrere KI-Modelle mithilfe des Satbilder-Datensatzes trainiert.   
Das Training wurde mithilfe des [Diffusers Repository](https://github.com/huggingface/diffusers) durchgeführt.   
Eine Anleitung zum Training ist im Abschnitt [KI-Modell trainieren](#-ki-modell-trainieren), bzw. in [/model_training/model_training.ipynb](https://github.com/GP-Alternativweltgeschichten/AI_Code/blob/master/Model_Training/model_training.ipynb) zu finden.   
Alle Modelle basieren auf dem [Stable Diffusion v2-1 Modell](https://huggingface.co/stabilityai/stable-diffusion-2-1).   
Die aktuellen Modelle sind auf Huggingface zu finden:   
- [OlpeAI](https://huggingface.co/GP-Alternativweltgeschichten/OlpeAI) (10k Schritte)   
- [OlpeAI_Small](https://huggingface.co/GP-Alternativweltgeschichten/OlpeAI_Small) (700 Schritte)   

## ⚙️ Voraussetzungen
1. Installieren Sie eine Python-Entwicklungsumgebung und ein Tool zum Verwalten von Python-Umgebungen (Optional, aber _sehr_ hilfreich).   
   In diesem Projekt wurde dazu [Pycharm](https://www.jetbrains.com/de-de/pycharm/) und [Anaconda](https://www.anaconda.com/download) verwendet.
3. Klonen Sie dieses Repository in Ihr Projektverzeichnis
4. Erstellen Sie eine Python-Umgebung mit Anaconda.  
   Dazu gehen Sie in Pycharm auf `File --> Settings --> Project: "Name des Projekts" --> Python Interpreter`.  
   Wählen dann Add `Interpreter --> Add Local Interpreter`.   
   Im neuen Fenster wählen Sie dann `Conda Environment` und `Create new environment`.   
   Als `Environment name` wählen Sie einen passenden Namen (z.B.: OlpeAI) und als `Python version: 3.10`.   
5. Öffnen Sie nun in Pycharm ein Terminal.   
   Vor dem Dateipfad sollte nun der Name der Anaconda-Umgebung stehen.   
   Ist dies nicht der Fall, kann mit dem folgenden Befehl die Anaconda-Umgebung aktiviert werden:
   ```sh
   conda activate <Name der Umgebung>
   ```
7. Installieren Sie in dem Terminal alle relevanten Python-Bibliotheken.   
   Verwenden Sie dazu die Befehle:
   ```sh
   pip install fastapi uvicorn pydantic diffusers pillow torch requests OpenAI deep_translator transformers
   
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
   ```
   Für die Pytorch Bibliotheken (zweiter Befehl) ist die im Projekt verwendete Version angegeben.   
   Ein Befehl zur Installation einer neueren Version kann auf der [Pytorch Website](https://pytorch.org/get-started/locally/) gefunden werden.   

## 🚀 Server ausführen
Erfüllen Sie zunächst die [Voraussetzungen](#-voraussetzungen).   
Starten Sie den KI-Server im Terminal (Anaconda-Umgebung muss aktiviert sein):
```sh
python ./server/inpaint_REST.py
```
Der Server läuft standardmäßig unter http://localhost:8000/

## ✨ Server Funktionalitäten
Welche Funktionen haben wir? Warum? Reihenfolge der Ausführung? Was kommt rein/Was geht raus? 

## 📝 KI-Prompting
Parameter? Was müssen wir beachten?

## 📥 KI-Modell trainieren
1. Erfüllen Sie zunächst die [Voraussetzungen](#-voraussetzungen).
2. Öffnen Sie nun [/model_training/model_training.ipynb](https://github.com/GP-Alternativweltgeschichten/AI_Code/blob/master/Model_Training/model_training.ipynb).
3. Führen Sie der Reihe nach alle Zellen des Jupyter Notebooks aus:
4. **Requirements:**
   1. Zunächst werden weitere, wichtige Bibliotheken installiert.
   2. Dann wird der im Projekt erstellte Datensatz ([Satbilder](https://huggingface.co/datasets/GP-Alternativweltgeschichten/Satbilder)) geladen.
   3. Nun werden die nötigen Bibliotheken geladen.
5. **Training the model:**   
   1. Dieser Schritt kann entweder direkt im Jupyter Notebook durchgeführt werden oder im Terminal.  
      Aus eigener Erfahrung ist das Ausführen des Befehls im Terminal empfehlenswert, da PyCharm nach einiger Zeit abstürzen kann.
      Achten Sie beim Ausführen im Terminal darauf, dass die Anaconda-Umgebung aktiviert ist.
   2. Passen Sie den Befehl auf die eigenen Bedürfnisse an.  
      Die wichtigsten Parameter sind dabei `train_batch_size`, `gradient_accumulation_steps` und `max_train_steps`.
      - **`train_batch_size`** bestimmt, wie viele Bilder gleichzeitig verarbeitet werden und hat den größten Einfluss auf den VRAM-Verbrauch.  
      - **`gradient_accumulation_steps`** gibt an, wie viele Schritte Gradienten gespeichert werden, bevor ein Update des Modells erfolgt.  
        Ein höherer Wert kann helfen, mit kleineren `train_batch_size`-Werten zu arbeiten, um VRAM zu sparen.  
      - **`max_train_steps`** definiert die maximale Anzahl an Trainingsschritten. Ein höherer Wert führt zu einer längeren Trainingszeit, _kann_ aber zu besseren Ergebnissen führen.
      
      Zudem kann das Modell, das als Grundlage verwendet wird, in `pretrained_model_name_or_path` angegeben werden.  
      Dabei kann entweder ein lokaler Pfad angegeben werden oder der Name des Modells auf [Hugging Face](https://huggingface.co/).  
   3. Führen Sie die Zelle, bzw. den Befehl im Terminal aus. Das Training sollte nun automatisch laufen.  
      Das Modell wird schließlich in `output_dir` gespeichert.
6. **Testing the model:**
   1. Zunächst wird das Modell in eine `StableDiffusionInpaintPipeline` geladen, die alle nötigen Funktionen bereitstellt.  
      Passen Sie dabei den Pfad (erster Parameter) auf den Namen des Modells an.
   2. Diese Pipeline wird dann in die GPU geladen.  
      Verwenden Sie keine GPU, so kann auch `cpu`, statt `cuda` angegeben werden.
   3. Nun werden die beiden grundlegenden Bilder geladen und angezeigt.  
      Der Bildausschnitt, der Veränderungen erhalten soll (`initial_image`) und die Maske (`mask_image`), die bestimmt wo die Veränderungen stattfinden.
   4. Danach können mit dem Modell Bilder generiert werden.  
      Dazu muss ein Prompt angegeben werden und dieser, zusammen mit den Bildern und weiteren Parametern an die Pipeline gegeben werden.
      Die Erklärung der einzelnen Parameter finden Sie in (//TODO).
   5. Schließlich werden die Bilder angezeigt. Es empfiehlt sich mehrere Bilder zu erzeugen, um die Konsistenz der Generierung zu überprüfen.



## 🔭 Ausblick
Nach umfassender Analyse und Evaluation wurde festgestellt, dass das KI-Modell in seiner gegenwärtigen Konfiguration noch nicht die gewünschte Trainingsreife aufweist. Für die Zukunft ist demnach die weitere Optimierung des Modells auf Basis einer größeren Anzahl noch detaillierter beschrifteter Bilder vorgesehen.
Darüber hinaus ist die Verwendung eines alternativen Modells für die gleiche Aufgabe eine mögliche zukünftige Entwicklung, sollte sich dieses als überlegen erweisen.
Darüber hinaus sollte, unter Beibehaltung der Topografie, eine Integration der Topografie in das KI-Modell erwogen werden. Dafür müsste das Modell eventuell neu trainiert werden oder gar ein anderes genutzt werden. Die Integration von Flüssen, die den Berg hinabfließen oder Seen, die auf reaktive Weise auf Bergen durch das Modell generiert werden, ist eine vielversprechende Möglichkeit, die durch die Einbindung der Topografie in das KI-Modell realisiert werden könnte. 
Für die Zukunft sehen wir auch die Möglichkeit, andere Modelle als LORAS zu verwenden, um die Stadt in verschiedenen Stilen wie Mittelalter, Japanisch, Wilder Westen usw. darzustellen.
Es ist jedoch darauf hinzuweisen, dass auch ein gänzlich divergierendes Vorgehen in der Zukunft nicht ausgeschlossen werden kann. Es existieren bereits ähnliche Projekte, wie das von Nvidia ([Nvidia Canvas](https://support.nvidia.eu/hc/de/articles/360017442939-NVIDIA-CANVAS)), die demonstrieren, dass die zugrunde liegende Problemstellung nicht ausschließlich durch die vorgeschlagene Lösung gelöst werden kann. Sollte sich für ein alternatives Vorgehen entschieden werden, wäre eine entsprechende Anpassung des Modells erforderlich.

## 🚮 Verworfene Features 
Die Mehrzahl der Features, die im Rahmen des Projektes konzipiert wurden, wurde in der vorliegenden Form realisiert. Dies ist auf die Tatsache zurückzuführen, dass sich die Projektgruppe bei der Entwicklung des KI-Modells nicht mit utopischen Zielen und Features befasst hat. 
Einige wenige Features wurden jedoch verworfen.
Ein Beispiel ist die Möglichkeit, die Text-to-Image-Funktion des Modells zu nutzen.
Diese wurde zunächst implementiert, dann jedoch durch die Inpaint-Funktion ersetzt. Dieser Schritt wurde unternommen, da die Notwendigkeit bzw. der Nutzen dieser Funktion für die Gäste nicht länger als gegeben angesehen wurde. Sollte eine Revision dieses Vorgehens in Betracht gezogen werden, wäre eine erneute Implementierung dieses Features eine mögliche Konsequenz.
Zudem wurde zu Projektbeginn die Idee diskutiert, die Karte in einer 3D-Ansicht zu präsentieren, in der das Modell darauf trainiert werden oder ein anderes Modell benötigt werden würde. Diese Idee wurde jedoch aus Gründen der Projektgröße verworfen. Darüber hinaus wurden weitere kleinere Features, die ähnliche Gründe wie Text-to-Image aufwiesen, entfernt.
Auch die Idee, die Stadt mit Hilfe von KI in verschiedenen Stilen wie Mittelalter oder wilder Westen zu gestalten, wurde zunächst zurückgestellt und aus Zeitgründen erstmals in den Ausblick für die zukünftige Entwicklung aufgenommen.

## 📂 Projektstruktur
```
/client                     # Verzeichnis für den Test-Client
  /rest_client.py           # Python-Datei für den Test-Client
/model_training             # Verzeichnis für das Trainieren und Testen von KI-Modellen
  /test_images              # Enthält zwei Bilder zum Testen der Inpaint-Funktion von KI-Modellen
  /model_training.ipynb     # Pyhton-Notebook mit allen relevanten Anforderungen und Skripten zum Trainieren und Testen von KI-Modellen
/persistence                # Verzeichnis zum Aufzeichnen der Prompts und Ausgewählten Bereiche der Nutzer
  /img                      # Verzeichnis zum Speichern der Eingabe-Bilder und generierten Ausgabe-Bilder
  /prompt                   # Verzeichnis zum Speichern der textuellen Prompts
/server                     # Verzeichnis für die Server-Dateien
  /image_processing.py      # Python-Datei mit Funktionen zur Verarbeitung der Bilder und dem Ausführen des Inpaintings
  /inpaint_REST.py          # Python-Datei für den Server
  /persistence.py           # Python-Datei zur Persistierung der Nutzereingaben und Ausgaben des KI-Modells
  /prompt_engineering.py    # Python-Datei mit einer Funktion zum Verbessern von Prompts
  /request_types.py         # Python-Datei, die die Form des Requests festlegt und Funktionen zum Verarbeiten dessen bereitstellt.
```

--- 
