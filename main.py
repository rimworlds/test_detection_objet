import cv2
import numpy as np
import tensorflow as tf

def preprocess(image):
    # Redimensionne l'image à la taille d'entrée attendue par le modèle
    input_shape = (300, 300)  # Remplace cela par la taille d'entrée de ton modèle
    resized_image = cv2.resize(image, input_shape)

    # Normalise les valeurs de pixel entre 0 et 1
    normalized_image = resized_image / 255.0

    # Convertit le type de données en UINT8
    input_data = (normalized_image * 255).astype(np.uint8)

    # Ajoute une dimension pour correspondre à la forme d'entrée du modèle
    input_data = np.expand_dims(input_data, axis=0)

    return input_data

def postprocess(output_data):

    # Assure-toi que la forme de output_data est celle attendue
    assert len(output_data.shape) == 3, "La forme de output_data n'est pas celle attendue"

    # Tu peux ajuster le reste de cette fonction en fonction des besoins de ton modèle
    detection_threshold = 0.5
    detected_boxes = output_data[0]  # Accède au premier tableau dans la dimension 0

    # Filtrer les détections en fonction du seuil de confiance
    valid_detections = detected_boxes[:, 2] >= detection_threshold

    # Retourner les résultats filtrés
    return {
        'boxes': detected_boxes[valid_detections],
        # Ajoutez d'autres champs nécessaires en fonction de la sortie de votre modèle
    }

def draw_results(image, results):
    # Assure-toi que 'boxes' existe dans les résultats
    if 'boxes' in results:
        boxes = results['boxes']
        
        # Dessine des rectangles autour des objets détectés
        for box in boxes:
            ymin, xmin, ymax, xmax = box
            xmin, xmax, ymin, ymax = int(xmin * image.shape[1]), int(xmax * image.shape[1]), int(ymin * image.shape[0]), int(ymax * image.shape[0])
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    return image

# Charger le modèle TensorFlow Lite
interpreter = tf.lite.Interpreter(model_path="model/sauvegarde.tflite")
interpreter.allocate_tensors()

# Obtient les détails de l'entrée du modèle
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

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