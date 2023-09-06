import numpy as np
from numpy import histogram


def evolucion_diferencial(NP, img, F=0.8, umbrales=2, num_generaciones=1, GR=0.8):
    # se crea la poblacion con distintos umbrales
    poblacion = np.array([np.random.randint(0, 255, umbrales) for _ in range(NP)])
    frecuencias, valores = np.histogram(img, bins=range(0, 256))
    valores = valores[:-1]
    Pi = frecuencias / len(img)
    i_Pi = np.multiply(Pi, valores)
    for generacion in range(num_generaciones):
        vectores_aleatorios = mutacion(NP, poblacion, F)
        vectores_prueba = recombinacion(NP, poblacion, vectores_aleatorios, GR)
        varianza_poblacion = calc_varianza(img, poblacion, Pi, i_Pi)
        varianza_vectores_prueba = calc_varianza(img, vectores_prueba, Pi, i_Pi)
        menor_varianza = varianza_vectores_prueba > varianza_poblacion
        poblacion = np.where(menor_varianza[:, None], vectores_prueba, poblacion)
    varianza_final = calc_varianza(img, poblacion, Pi, i_Pi)
    optimo = poblacion[np.argmax(varianza_final)]
    return np.sort(optimo)[::-1]


def mutacion(NP, poblacion, F):
    vectores_aleatorios = np.empty((NP, poblacion.shape[1]))
    for i, _ in enumerate(poblacion):
        # se eligen los vectores para crear los random noise vectors
        vectores_objetivo = poblacion[np.random.choice(NP, size=3, replace=False)]
        vector_aleatorio = F * (vectores_objetivo[0] - vectores_objetivo[1]) + vectores_objetivo[2]
        # acotamos los vectores creados y luego los redondeamos
        vector_aleatorio = np.clip(vector_aleatorio, 0, 255)
        vector_aleatorio = np.round(vector_aleatorio).astype(int)
        vectores_aleatorios[i] = vector_aleatorio
    return vectores_aleatorios


def recombinacion(NP, poblacion, vectores_aleatorios, GR):
    vector_recombinacion = np.random.rand(NP, poblacion.shape[1])
    vector_recombinacion = np.where(vector_recombinacion <= GR, 1, 0)
    vectores_prueba = np.where(vector_recombinacion == 1, vectores_aleatorios, poblacion)
    return vectores_prueba


def calc_varianza(img, umbrales, Pi, i_Pi):
    varianza_poblacion = np.zeros(len(umbrales))
    # Se calcula peso y media de gris por umbral en la imagen
    umbrales_ordenados = np.sort(umbrales, axis=1)
    for i, individuo_umbrales in enumerate(umbrales_ordenados):
        umbral_anterior = 0
        peso = np.zeros(len(individuo_umbrales) + 1)
        media_gris = np.zeros(len(individuo_umbrales) + 1)
        for j, umbral in enumerate(individuo_umbrales):
            media_gris[j] = np.sum(i_Pi[int(umbral_anterior):int(umbral)])
            peso[j] = np.sum(Pi[int(umbral_anterior):int(umbral)])
            umbral_anterior = umbral

        peso[-1] += np.sum(Pi[int(individuo_umbrales[-1]):])
        media_gris[-1] += np.sum(i_Pi[int(individuo_umbrales[-1]):])
        media_gris = np.divide(media_gris, peso, out=np.zeros_like(media_gris), where=peso != 0)
        intensidad_promedio_img = np.sum(peso * media_gris)
        varianza_poblacion[i] = np.sum(peso * (media_gris - intensidad_promedio_img) ** 2)
    return varianza_poblacion
