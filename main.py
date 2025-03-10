import gradio as gr
import numpy as np


def tuo_cognome_nome(image: np.ndarray) -> np.ndarray:
    """Esempio di funzione che processa una immagine, generando un'altra immagine"""
    return image


if __name__ == '__main__':
    gr.Interface(
        fn=tuo_cognome_nome,
        inputs=['image'],
        outputs=['image']
    ).launch()
