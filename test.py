import os
import numpy as np
import cv2
import tensorflow as tf
import gradio as gr
import mediapipe as mp

#Percorso del modello addestrato
dataset_path = "/Users/michelepotsios/Desktop/progetto rilevamento segni /"
model_save_path = os.path.join(dataset_path, "sign_language_model.h5")

# Caricamento del modello salvato**
if not os.path.exists(model_save_path):
    raise FileNotFoundError(f"Il file {model_save_path} non esiste. Controlla il percorso!")

model = tf.keras.models.load_model(model_save_path)
print("✅ Modello caricato con successo!")

# inizializzazione di MediaPipe per il rilevamento delle mani
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Mappa delle lettere della lingua dei segni**
label_map = {i: chr(65 + i) for i in range(25)}  # 0=A, 1=B, ..., 24=Y

#Funzione di preprocessing dell'immagine
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))
    normalized = resized / 255.0
    return np.reshape(normalized, (1, 28, 28, 1))

#Funzione per il riconoscimento in tempo reale
def recognize_live():
    cap = cv2.VideoCapture(0)  # Attiva la webcam

    with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            label = "Nessun segno rilevato"
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    h, w, _ = frame.shape
                    x_min, x_max = int(min([lm.x for lm in hand_landmarks.landmark]) * w), int(max([lm.x for lm in hand_landmarks.landmark]) * w)
                    y_min, y_max = int(min([lm.y for lm in hand_landmarks.landmark]) * h), int(max([lm.y for lm in hand_landmarks.landmark]) * h)

                    roi = frame[y_min:y_max, x_min:x_max]

                    if roi.size > 0:
                        processed = preprocess_image(roi)
                        prediction = model.predict(processed)
                        class_idx = np.argmax(prediction)
                        confidence = prediction[0][class_idx]

                        if confidence > 0.8:
                            label = label_map[class_idx]
                            cv2.putText(frame, f"Letter: {label}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            yield frame[:, :, ::-1], label  # Converti BGR → RGB per Gradio

        cap.release()
        cv2.destroyAllWindows()

# Avvio di Gradio per la visualizzazione live
gr.Interface(
    fn=recognize_live,
    inputs=[],
    outputs=["image", "text"],
    live=True,
    title="📷 Riconoscimento della Lingua dei Segni in Tempo Reale",
    description="Accendi la webcam e mostra un segno per vedere il riconoscimento in tempo reale!"
).launch()