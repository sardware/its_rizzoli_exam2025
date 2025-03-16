import cv2 as cv
import numpy as np
import random
import gradio as gr

# forme del mosaico selezionabili con True/False
shape_settings = {"circle": True,
                  "square": True,
                  "rectangle": True,
                  "triangle": True, 
                  "hexagon": True}

image_path = "captured_image.png"

# funzione scatta foto (premi y per scattare)
def cattura_imagine():
    cap = cv.VideoCapture(0) 
    
    while True:
        _, frame = cap.read() 

        frame = cv.flip(frame, 1) 
        cv.imshow("Premi 'Y' per scattare", frame) 

        key = cv.waitKey(1)
        if key & 0xFF == ord('y'):  
            cv.imwrite(image_path, frame)  
            break

    cap.release()
    cv.destroyAllWindows() 
    return image_path

# funzione genera mosaico
def sassi_emanuele(image, 
                   max_size=32, 
                   border_thickness=1, 
                   sobel_threshold=50, 
                   window_sizes=[32, 16, 8],
                   step_div=2,
                   min_draw_size=8,
                   shape_settings=shape_settings):
    
    # funzione per colore bordo scuro
    def scurisci_bordo(color, amount=40):
        
        # riceve in ingresso i colori medi RGB e sottrae 40, se i valori sono < 40, torna 0
        dark_color = [max(c - amount, 0) for c in color]
        return dark_color
    
    # funzione per disegnare forma randomica
    def disegna_shape(mosaic, original, x, y, w_size, height, width, shape_settings):
        
        # se la finestra è più piccola della dimensione MINIMA delle forme, non deve restituire nulla.
        if w_size < min_draw_size:
            return
        
        # determino punto in basso a destra dell'area dalla quale voglio considerare i valori 'medi' del colore
        x2 = min(x + w_size, width)
        y2 = min(y + w_size, height)

        # estraggo una sottomatrice dell'immagine, calcolo i valori medi e li riporto come una lista di interi
        avg_color = np.mean(original[y:y2, x:x2], axis=(0,1)).astype(int).tolist()
        border_color = scurisci_bordo(avg_color)

        # filtro le forme in base a quelle selezionate nel dizionario
        available_shapes = []
        for shape, enabled in shape_settings.items():
            if enabled:
                available_shapes.append(shape)

        # se sono state selezionate forme vengono selezionate random altrimenti non viene disegnato nulla
        if available_shapes:
            shape_type = random.choice(available_shapes)
        else:
            return
        
        # disegno cerchio 
        if shape_type == "circle":

            # raggio
            r = w_size // 2

            # centro (sommo il raggio ad entrambe le coordinate del punto in alto a sx)
            center = (x + r, y + r)

            # disegno il cerchio sull'immagine nuova
            cv.circle(mosaic, center, r, avg_color, -1)

            # disegno il bordo 
            cv.circle(mosaic, center, r, border_color, border_thickness)

        # disegno quadrato
        elif shape_type == "square":

            # disegno quadrato sull'immagine nuova
            cv.rectangle(mosaic, (x, y), (x2, y2), avg_color, -1)

            # disegno il bordo
            cv.rectangle(mosaic, (x, y), (x2, y2), border_color, border_thickness)

        # disegno rettangolo
        elif shape_type == "rectangle":

            # larghezza e altezza rettangolo (randomiche e comprese tra w_size e la sua metà)
            w_rect = random.randint(w_size // 2, w_size)
            h_rect = random.randint(w_size // 2, w_size)

            # calcolo le coordinate del punto di 'inizio' del rettangolo (in modo che questo non esca dalla finestra)
            rx = x + random.randint(0, max(0, w_size - w_rect))
            ry = y + random.randint(0, max(0, w_size - h_rect))

            # calcolo estremo in basso a dx
            rx2 = min(rx + w_rect, width)
            ry2 = min(ry + h_rect, height)

            # disegno rettangolo sull'immagine nuova
            cv.rectangle(mosaic, (rx, ry), (rx2, ry2), avg_color, -1)
            
            # disegno il bordo
            cv.rectangle(mosaic, (rx, ry), (rx2, ry2), border_color, border_thickness)

        # disegno triangolo
        elif shape_type == "triangle":

            # calcolo posizione vertice centrale 
            half = w_size // 2

            # creo array dei tre vertici del triangolo
            ver = np.array([(x + half, y), (x, y + w_size), (x + w_size, y + w_size)])

            # disegno il triangolo sull'immagine nuova
            cv.drawContours(mosaic, [ver], 0, avg_color, -1)

            # disegno i contorni
            cv.drawContours(mosaic, [ver], 0, border_color, border_thickness)
        
        # disegno esagono
        elif shape_type == "hexagon":

            # definisco il raggio
            r = w_size // 2

            # creo un array coi vertici dell'esagono
            ver = np.array([(
                int(x + r + r * np.cos(np.deg2rad(60 * i))),
                int(y + r + r * np.sin(np.deg2rad(60 * i)))
            ) for i in range(6)] )

            # disegno esagono sull'immagine nuova
            cv.drawContours(mosaic, [ver], 0, avg_color, -1) 

            # disegno il bordo
            cv.drawContours(mosaic, [ver], 0, border_color, border_thickness)

    # altezza e larghezza immagine
    height, width = image.shape[:2]

    # creo nuova immagine (bianca)
    mosaic = np.full_like(image, 255)

    # individuo i punti bianchi
    empty_spots = np.where(np.all(mosaic == 255, axis=-1))

    # creo una lista con le posizioni dei punti bianchi
    empty_positions = list(zip(empty_spots[0], empty_spots[1]))
    
    # randomizzo l'ordine delle posizioni dei punti
    random.shuffle(empty_positions)

    # aggiungo forme random per coprire le aree vuote
    for (y, x) in empty_positions:

        # determino una dimensione casuale per la forma
        size = random.randint(min_draw_size, max_size)
        
        # disegno la forma
        disegna_shape(mosaic, image, x, y, size, height, width, shape_settings)

    # SOBEL
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
    sobely = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobelx**2 + sobely**2)
    _, sobel_mask = cv.threshold(np.uint8(np.clip(sobel, 0, 255)), sobel_threshold, 255, cv.THRESH_BINARY)

    # ciclo su tutte le dimensioni delle finestre scelte
    for w_size in window_sizes:

        # se la dimensione è minore della forma minima non disegna niente
        if w_size < min_draw_size:
            continue

        # definisco lo step tra una forma e l'altra in base al massimo tra (dimensione // step_div) ed 1
        step = max(w_size // step_div, 1)

        # calcolo la lista delle 'posizioni' disponibili e 'rimescolo' in modo che le forme vengano disegnate in modo casuale
        available_positions = [(x, y) for y in range(0, height, step) for x in range(0, width, step)]
        random.shuffle(available_positions)

        # applico il filtro di Sobel (per catturare più dettagli ai bordi)
        for (x, y) in available_positions:

            # controllo se nella finestra c'è almeno un bordo (trovato con Sobel), in quel caso richiamo la funzione disegna_shape e disegno una forma
            if np.any(sobel_mask[y:min(y + w_size, height), x:min(x + w_size, width)] > 0):
                disegna_shape(mosaic, image, x, y, w_size, height, width, shape_settings)

    return mosaic

# immagine da trattare
captured_image = cattura_imagine()

# app gradio
if __name__ == '__main__':
    gr.Interface(
        fn=sassi_emanuele,  
        inputs=gr.Image(type="numpy", value=captured_image),  
        outputs=gr.Image(type="numpy")  
).launch(inbrowser=True)
