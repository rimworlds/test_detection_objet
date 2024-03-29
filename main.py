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

    # Tu peux ajuster le reste de cette fonction en fonction des besoins de ton modèle
    detection_threshold = 0.8
    detected_boxes = output_data[0]  # Accède au premier tableau dans la dimension 0
    detected_classes = output_data[0]
    detected_scores = output_data[0]

    # Filtrer les détections en fonction du seuil de confiance
    valid_detections = detected_boxes[:, 2] >= detection_threshold

    # Retourner les résultats filtrés
    return {
        'boxes': detected_boxes[valid_detections],
        'classes': detected_classes[valid_detections],
        'scores': detected_scores[valid_detections],
        
        # Ajoutez d'autres champs nécessaires en fonction de la sortie de votre modèle
    }

def draw_results(image, results, class_labels):
    # Assure-toi que 'boxes', 'classes' et 'scores' existent dans les résultats
    if 'boxes' in results and 'classes' in results and 'scores' in results:
        boxes = results['boxes']
        classes = results['classes']
        scores = results['scores']
        
        # Dessine des rectangles autour des objets détectés avec les classes et les scores
        for box, class_id, score in zip(boxes, classes, scores):
            ymin, xmin, ymax, xmax = box
            xmin, xmax, ymin, ymax = int(xmin * image.shape[1]), int(xmax * image.shape[1]), int(ymin * image.shape[0]), int(ymax * image.shape[0])
            
            # If class_id is an array, use class_id[0] or iterate through its elements
            if np.isscalar(class_id):
                class_label = class_labels[int(class_id)]
            else:
                class_label = [class_labels[int(cid)] for cid in class_id]

            if np.isscalar(score):
                label = f"{class_label}: {score:.2f}"
            else:
                label = ', '.join([f"{cl}: {sc:.2f}" for cl, sc in zip(class_label, score)])

            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return image

# Charger le modèle TensorFlow Lite
interpreter = tf.lite.Interpreter(model_path="model/sauvegarde.tflite")
interpreter.allocate_tensors()

# Obtient les détails de l'entrée du modèle
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_labels = ["class1", "class2", "class3"]

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
    frame = draw_results(frame, results, class_labels)
    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()