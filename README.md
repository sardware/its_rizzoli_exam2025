# Rilevamento Volti, Contorno e Inversione Colori

Questa applicazione Gradio rileva i volti in un'immagine, crea un contorno di puntini rossi attorno ai volti rilevati e inverte i colori all'interno del contorno. Utilizza OpenCV per il rilevamento dei volti tramite il classificatore a cascata `haarcascade_frontalface_default.xml`.

## Funzionalità

*   Rilevamento di volti in immagini.
*   Creazione di un contorno di puntini rossi attorno ai volti rilevati.
*   Inversione dei colori all'interno del contorno.
*   Supporta input da webcam, upload di immagini locali e copia/incolla dalla clipboard.

## Requisiti

*   Python 3.6+
*   Librerie Python elencate in `requirements.txt` (vedi sotto).

## Installazione

1.  Clona questo repository o scarica il file `main.py`.
2.  Assicurati di avere Python installato.
3.  Crea un ambiente virtuale (consigliato):

    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    venv\Scripts\activate.bat  # Windows
    ```

4.  Installa le dipendenze:

    ```bash
    pip install -r requirements.txt
    ```


## Esecuzione

1.  Esegui l'applicazione:

    ```bash
    python main.py
    ```

2.  Apri il link fornito nel terminale nel tuo browser web.
3.  Carica un'immagine, utilizza la webcam o incolla un'immagine dalla clipboard.
4. Clicca su Submit e verifica il risultato.

## Note

*   Il rilevamento dei volti potrebbe non essere perfetto e potrebbe non funzionare su tutte le immagini.
*   Il classificatore a cascata `haarcascade_frontalface_default.xml` rileva solo i volti frontali.

## Autore

Emanuele Miranda