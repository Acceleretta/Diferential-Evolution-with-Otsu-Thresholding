import matplotlib.pyplot as plt
import cv2
import Evolucion_Diferencial as ed
import numpy as np


# Se convierte la imagen en gris al solo tomar un canal del RGB
img = cv2.imread("prueba.jpg", 0)
umbrales = ed.evolucion_diferencial(150, img, umbrales=3, num_generaciones=15)
indices = np.digitize(img, umbrales) - 1
valores_redondeados = umbrales[indices]
print(umbrales)
plt.imshow(valores_redondeados, cmap='gray')  # Especificar cmap='gray'
plt.show()

