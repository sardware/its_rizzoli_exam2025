import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageChops
from scipy.spatial import Voronoi
import random
import cv2

# Il programma funziona, a volte se il soggetto non è ben delineato
# non si riesce a riconoscere, ma con soggetti più marcati da un bel risultato

def voronoi_finite_polygons_2d(vor, radius=None):
    nuove_regioni = []
    nuovi_vertici = vor.vertices.tolist()

    centro = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max() * 2

    # Mappa tutte le creste per ogni punto
    tutte_creste = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        tutte_creste.setdefault(p1, []).append((p2, v1, v2))
        tutte_creste.setdefault(p2, []).append((p1, v1, v2))

    # Ricostruzione delle regioni
    for p1, regione in enumerate(vor.point_region):
        vertici_correnti = vor.regions[regione]
        if all(v >= 0 for v in vertici_correnti):
            nuove_regioni.append(vertici_correnti)
            continue

        creste = tutte_creste[p1]
        nuova_regione = [v for v in vertici_correnti if v >= 0]

        for p2, v1, v2 in creste:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue
            # Impone un limite di dimensione per evitare poligoni troppo grandi
            # In questo modo si ha una immagine sempre dettagliata
            t = vor.points[p2] - vor.points[p1]
            t = t / np.linalg.norm(t)
            n = np.array([-t[1], t[0]])
            punto_lontano = vor.vertices[v2] + n * radius
            nuova_regione.append(len(nuovi_vertici))
            nuovi_vertici.append(punto_lontano.tolist())
        nuove_regioni.append(nuova_regione)
    return nuove_regioni, np.asarray(nuovi_vertici)

def clip_polygon(polygon, bbox):
    def clip(soggetto, lato):
        tagliato = []
        for i in range(len(soggetto)):
            corrente = soggetto[i]
            precedente = soggetto[i - 1]
            if dentro(corrente, lato):
                if not dentro(precedente, lato):
                    intersezione = intersezione_tra(precedente, corrente, lato)
                    if intersezione is not None:
                        tagliato.append(intersezione)
                tagliato.append(corrente)
            elif dentro(precedente, lato):
                intersezione = intersezione_tra(precedente, corrente, lato)
                if intersezione is not None:
                    tagliato.append(intersezione)
        return tagliato

    def dentro(punto, lato):
        x, y = punto
        xmin, ymin, xmax, ymax = bbox
        if lato == 'left':   return x >= xmin
        if lato == 'right':  return x <= xmax
        if lato == 'top':    return y >= ymin
        if lato == 'bottom': return y <= ymax
        return True

    def intersezione_tra(p1, p2, lato):
        x1, y1 = p1
        x2, y2 = p2
        xmin, ymin, xmax, ymax = bbox
        if x1 == x2 and y1 == y2:
            return p1
        if lato == 'left':
            x = xmin
            if x2 - x1 != 0:
                y = y1 + (y2 - y1) * (xmin - x1) / (x2 - x1)
                return (x, y)
        elif lato == 'right':
            x = xmax
            if x2 - x1 != 0:
                y = y1 + (y2 - y1) * (xmax - x1) / (x2 - x1)
                return (x, y)
        elif lato == 'top':
            y = ymin
            if y2 - y1 != 0:
                x = x1 + (x2 - x1) * (ymin - y1) / (y2 - y1)
                return (x, y)
        elif lato == 'bottom':
            y = ymax
            if y2 - y1 != 0:
                x = x1 + (x2 - x1) * (ymax - y1) / (y2 - y1)
                return (x, y)
        return None

    poligono = polygon
    for lato in ['left', 'right', 'top', 'bottom']:
        poligono = clip(poligono, lato)
    return poligono

def polygon_area(poly):
    if len(poly) < 3:
        return 0
    area = 0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2

def stained_glass_precise(image: np.ndarray) -> np.ndarray:
    #Crea una immagine con effetto mosaico, ho preso ispirazione dalle vetrate delle chiese
    if image is None:
        return None

    # Converto l'immagine in un oggetto PIL
    im = Image.fromarray(image)
    width, height = im.size
    area = width * height
    max_area_poligono = 3000  # Area massima desiderata per un poligono (in pixel)

    # Usa OpenCV per il rilevamento preciso dei bordi
    image_cv = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2GRAY)
    bordi = cv2.Canny(image_cv, threshold1=50, threshold2=150)

    # Campiono alcuni punti dai bordi (solo una frazione per evitare overfitting)
    indici_bordi = np.argwhere(bordi > 0)
    n_punti_bordo = int(len(indici_bordi) * 0.2)  # Uso il 20% dei punti rilevati
    if n_punti_bordo > 0:
        punti_bordo_campionati = indici_bordi[np.random.choice(len(indici_bordi), n_punti_bordo, replace=False)]
        # Conversione da (riga, colonna) a (x, y)
        punti_bordo = punti_bordo_campionati[:, [1, 0]]
    else:
        punti_bordo = np.empty((0, 2), dtype=int)

    # Genera punti seed distribuiti uniformemente
    n_uniform = max(200, int(area / max_area_poligono))
    punti_uniformi = np.column_stack((
        np.random.randint(0, width, n_uniform),
        np.random.randint(0, height, n_uniform)
    ))

    # Combina i punti uniformi e i punti dai bordi
    tutti_punti = np.concatenate((punti_uniformi, punti_bordo), axis=0)

    # Calcola il reticolo
    vor = Voronoi(tutti_punti)
    regioni, vertici = voronoi_finite_polygons_2d(vor, radius=1000)

    tela = Image.new("RGB", (width, height), (255, 255, 255))
    disegno = ImageDraw.Draw(tela)

    # Calcolo della luminosità media per distinguere il soggetto dallo sfondo
    luminosita_globale = np.mean(image_cv)
    im_np = np.array(im)
    bbox = (0, 0, width, height)

    for regione in regioni:
        poligono = vertici[regione]
        poly = [tuple(p) for p in poligono]
        poly = clip_polygon(poly, bbox)
        if len(poly) < 3:
            continue

        # Controlla che il poligono non superi troppo l'area desiderata
        area_poly = polygon_area(poly)
        if area_poly > max_area_poligono * 1.5:
            continue

        # Creo una maschera per il poligono
        mask_img = Image.new("L", (width, height), 0)
        ImageDraw.Draw(mask_img).polygon(poly, outline=1, fill=1)
        mask = np.array(mask_img)
        if np.sum(mask) == 0:
            continue

        # Calcolo del colore medio per il poligono
        r_avg = np.sum(im_np[:, :, 0] * mask) / np.sum(mask)
        g_avg = np.sum(im_np[:, :, 1] * mask) / np.sum(mask)
        b_avg = np.sum(im_np[:, :, 2] * mask) / np.sum(mask)
        colore_medio = (int(r_avg), int(g_avg), int(b_avg))
        # Aggiungo una leggera variazione casuale per dare sempre un risultato unico
        def varia(c):
            return int(max(0, min(255, c + random.randint(-10, 10))))
        colore_medio = (varia(colore_medio[0]), varia(colore_medio[1]), varia(colore_medio[2]))
        # Se il colore calcolato è troppo chiaro (quasi bianco), ne genero uno casuale
        # So che non risolve il problema, ma è un workaround che funziona per l'obiettivo che volevo raggiungere
        if colore_medio[0] > 240 and colore_medio[1] > 240 and colore_medio[2] > 240:
            colore_medio = (random.randint(0, 230), random.randint(0, 230), random.randint(0, 230))

        # Stimo se il poligono appartiene al soggetto
        centroide = (sum([p[0] for p in poly]) / len(poly), sum([p[1] for p in poly]) / len(poly))
        cx, cy = int(centroide[0]), int(centroide[1])
        regione_soggetto = False
        if 0 <= cx < width and 0 <= cy < height:
            luminosita_locale = image_cv[cy, cx]
            if luminosita_locale < luminosita_globale:
                regione_soggetto = True

        # Riempio il poligono e disegno il bordo
        disegno.polygon(poly, fill=colore_medio)
        larghezza_bordo = 4 if regione_soggetto else 2
        colore_bordo = (30, 30, 30) if regione_soggetto else (80, 80, 80)
        disegno.line(poly + [poly[0]], fill=colore_bordo, width=larghezza_bordo)

    # Applico una leggera sfocatura 
    tela = tela.filter(ImageFilter.GaussianBlur(radius=1))
    # Aggiungo una sovrapposizione lucida per simulare la diffusione della luce
    sovrapposizione = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    disegno_overlay = ImageDraw.Draw(sovrapposizione)
    for x in range(width):
        alpha = int(80 * (1 - abs(x - width / 2) / (width / 2)))
        disegno_overlay.line([(x, 0), (x, height)], fill=(255, 255, 255, alpha))
    tela = Image.alpha_composite(tela.convert("RGBA"), sovrapposizione).convert("RGB")

    return np.array(tela)
# Main
if __name__ == '__main__':
    demo = gr.Interface(
        fn=stained_glass_precise,
        inputs=gr.Image(label="Input Webcam", type="numpy"),
        outputs=gr.Image(label="Effetto Vetrata Artistica Preciso"),
        live=True
    )
    demo.launch()
