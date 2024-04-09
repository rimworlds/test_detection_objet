from tflite_model_maker import object_detector
from tflite_model_maker.object_detector import DataLoader

# Chemin vers le dossier principal contenant les sous-dossiers d'images
data_dir = 'chemin/vers/dataset'

# Créer un DataLoader à partir des images dans les sous-dossiers
data_loader = DataLoader.from_folder(data_dir)

# Split data into training and testing sets
train_data, test_data = data_dir.split(0.9)

# Train the model
model = object_detector.create(train_data)

# Evaluate the model
model.evaluate(test_data)

# Export the model to TFLite
model.export(export_dir='./model')