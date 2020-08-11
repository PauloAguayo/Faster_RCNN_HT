# Faster_RCNN_HT
En el presente trabajo se trató la detección y tracking de personas en condiciones no ideales, tales como medios de transportes públicos.

El algoritmo de tracking se ejecutó con el manejo de Kalman y linear assignment (algoritmo húngaro).
La gestión de detecciones se realizó con una red personalizada entrenada en Tensorflow. Se implementó Faster RCNN con inceptionV2.

La red utilizada es análoga al repositorio https://github.com/PauloAguayo/DensidadDePasajerosMovilidadReducida . La diferencia con este 
trabajo es que sólo se destinaron detecciones de cabezas. Por esta razón, las bases de datos utilizadas son iguales.

En la carpeta Faster_RCNN se encuentran tanto el modelo ".pb", como las etiquetas ".pbtxt".

En la carpeta "Results" se pueden encontrar resultados del desempeño del código.

# Instalación
Se debe generar un ambiente virtual con las siguientes librerías:

- Python >=3.0
- Tensorflow 1.14
- Numpy
- OpenCV
- Filterpy
- Numba
- scikit-learn==0.22.2

# Parser
El programa posee 6 variables parser, de las cuales sólo 3 son obligatorias. Se describen a continuación:

- '-m' (obligatoria): Ruta y nombre del modelo ".pb".
- '-l' (obligatoria): Ruta y nombre de las etiquetas o labels ".pbtxt".
- '-i' (obligatoria): Ruta y nombre del video.
- '-c' (opcional): Mínima probabilidad de detección. Default = 0.4.
- '-r' (opcional): Re-dimensionamiento de frames. Default=1080,640. Se debe seguir la misma estructura que en Default en caso de querer cambiar el tamaño.
- '-rec' (opcional): Ruta, nombre y formato del video de salida.

# Ejemplo
python main.py -m Faster_RCNN/frozen_inference_graph.pb -l Faster_RCNN/labelmap.pbtxt  -i GOPRO.mp4

# Agradecimientos
- https://github.com/ZidanMusk/experimenting-with-sort
- https://github.com/tensorflow/models
