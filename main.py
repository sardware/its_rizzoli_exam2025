import gradio as gr
import numpy as np
import cv2


def create_gradient_canvas(height, width, start_color=(173, 216, 230), end_color=(230, 190, 255)):
    """
    Questa funzione l'ho creata per creare una tela, come in un dipinto, a gradiente
    """
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    for x in range(width):
        alpha = x / width
        color = tuple(int((1 - alpha) * start + alpha * end) for start, end in zip(start_color, end_color))
        canvas[:, x] = color
    return canvas


def gambardella_vincenzo(image: np.ndarray,
                         threshold1: int = 100,
                         threshold2: int = 200,
                         draw_mode: str = "Circles",
                         color_mode: str = "Random Colors") -> np.ndarray:
    if image is None or image.size == 0:
        return create_gradient_canvas(300, 400)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1, threshold2)
    height, width = image.shape[:2]
    canvas = create_gradient_canvas(height, width)  # base sfondo gradiente creata sopra nella funzione
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if color_mode == "Random Colors":
            color = tuple(np.random.randint(0, 255, size=3).tolist())
        elif color_mode == "Gradient":
            avg_x = np.mean(contour[:, 0, 0])
            intensity = int((avg_x / width) * 255)
            color = (intensity, intensity, 255 - intensity)

        if draw_mode == "Circles":
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(canvas, center, radius, color, thickness=-1)
        elif draw_mode == "Rectangles":
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(canvas, (x, y), (x + w, y + h), color, thickness=-1)
        elif draw_mode == "Polygons":
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            cv2.fillPoly(canvas, [approx], color)
    return canvas


if __name__ == '__main__':
    interface = gr.Interface(
        fn=gambardella_vincenzo,
        inputs=[
            gr.Image(sources=["webcam", "upload"], type="numpy", label="Webcam o Carica Immagine"),
            gr.Slider(50, 300, value=100, step=10, label="Threshold 1 (Low)"),
            gr.Slider(50, 300, value=200, step=10, label="Threshold 2 (High)"),
            gr.Radio(["Circles", "Rectangles", "Polygons"], label="Modalità Disegno"),
            gr.Radio(["Random Colors", "Gradient"], label="Colorazione")
        ],
        outputs=gr.Image(type="numpy", label="Immagine Processata"),
        live=True,
        description="Ricordarsi di scattare la foto!"
    )
    interface.launch()
