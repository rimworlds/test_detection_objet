import cv2
import numpy as np
import tensorflow as tf

# Charger le modèle TensorFlow Lite
interpreter = tf.lite.Interpreter(model_path="/model/sauvegarde.tflite")
interpreter.allocate_tensors()

# Paramètres de la webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture de la vidéo depuis la webcam
    ret, frame = cap.read()

    # Prétraitement de l'image (ajuster la taille, normaliser, etc.)
    input_data = preprocess(frame)

    # Effectuer l'inférence avec le modèle
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Post-traitement des résultats (affichage des boîtes de détection, etc.)
    results = postprocess(output_data)

    # Afficher les résultats sur le flux vidéo en direct
    frame = draw_results(frame, results)
    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()