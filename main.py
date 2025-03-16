import gradio as gr
import numpy as np
import cv2

# Carica il classificatore a cascata per il rilevamento dei volti
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def miranda_emanuele(image: np.ndarray) -> np.ndarray:
    try:
        if image is None:
            print("Errore: Immagine vuota ricevuta.")
            return np.zeros((480, 640, 3), dtype=np.uint8)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Rileva i volti nell'immagine 
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Itera attraverso i volti rilevati
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                for i in range(x, x + w, 5):  # Itera lungo la larghezza del volto con un passo di 5 pixel
                    if y < image.shape[0] and i < image.shape[1]:   # Verifica che il punto sia all'interno dei limiti dell'immagine
                        image[y, i] = [0, 0, 255]  # Imposta il pixel superiore del contorno a rosso
                    if y + h - 1 < image.shape[0] and i < image.shape[1]:  # Verifica che il punto sia all'interno dei limiti dell'immagine
                        image[y + h - 1, i] = [0, 0, 255]  # Imposta il pixel inferiore del contorno a rosso
                for j in range(y, y + h, 5):  # Itera lungo l'altezza del volto con un passo di 5 pixel
                    if j < image.shape[0] and x < image.shape[1]:  # Verifica che il punto sia all'interno dei limiti dell'immagine
                        image[j, x] = [0, 0, 255]  # Imposta il pixel sinistro del contorno a rosso
                    if j < image.shape[0] and x + w - 1 < image.shape[1]:  # Verifica che il punto sia all'interno dei limiti dell'immagine
                        image[j, x + w - 1] = [0, 0, 255]  # Imposta il pixel destro del contorno a rosso

                # Inverti i colori all'interno del volto rilevato
                roi = image[y:y + h, x:x + w]  # Estrae la regione di interesse (ROI) che corrisponde al volto rilevato
                roi = cv2.bitwise_not(roi)  # Inverte i colori della ROI utilizzando l'operazione bitwise NOT
                image[y:y + h, x:x + w] = roi  # Sostituisce la ROI originale con la versione con i colori invertiti nell'immagine principale
        else:
            print("Nessun volto rilevato nell'immagine.")

        return image

    except Exception as e:
        print(f"Errore: {e}")
        return np.zeros((480, 640, 3), dtype=np.uint8)

# Configura e avvia l'interfaccia utente di Gradio per l'applicazione.
# L'interfaccia prende un'immagine come input (da webcam, upload o clipboard),
# applica la funzione 'miranda_emanuele' per rilevare volti, creare contorni di puntini rossi
# e invertire i colori all'interno dei volti rilevati, e mostra l'immagine elaborata come output.
if __name__ == '__main__':
    gr.Interface(
        fn=miranda_emanuele,
        inputs=gr.Image(sources=["webcam", "upload", "clipboard"], label="Input Immagine"),
        outputs=gr.Image(label="Immagine Elaborata"),
        title="Web App per detection del viso e inversione dei colori all'interno",
        description="Essa permette di carica un0immagine, di incollarne una o di usare la fotocamera, da essa rileva il viso, ed inverte i colori della Canny"
    ).launch()
