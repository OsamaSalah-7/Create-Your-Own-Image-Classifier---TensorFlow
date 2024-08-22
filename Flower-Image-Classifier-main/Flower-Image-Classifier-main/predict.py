import os
import json
import argparse
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub

def load_trained_model(model_directory):
    model = tf.keras.models.load_model(model_directory, custom_objects={'KerasLayer': hub.KerasLayer}, compile=False)
    return model

def preprocess_image(image_array):
    target_size = 224
    image_array = tf.image.resize(image_array, (target_size, target_size))
    image_array = tf.cast(image_array, tf.float32) / 255.0
    return image_array.numpy()

def make_prediction(image_file_path, trained_model, top_k_predictions=5):
    image = Image.open(image_file_path)
    image_as_array = np.asarray(image)
    processed_image = preprocess_image(image_as_array)
    processed_image = np.expand_dims(processed_image, axis=0)

    prediction_values = trained_model.predict(processed_image)[0]
    top_class_indices = np.argsort(-prediction_values)[:top_k_predictions]
    top_probabilities = prediction_values[top_class_indices]
    return top_probabilities, top_class_indices

def main():
    parser = argparse.ArgumentParser(description='Image Classification Prediction Tool')
    parser.add_argument('image_file', type=str, help='File path of the image to classify')
    parser.add_argument('model_file', type=str, help='File path of the pre-trained model')
    parser.add_argument('--top_k', type=int, default=5, help='Return the top K predicted classes')
    parser.add_argument('--label_map', type=str, required=True, help='File path to a JSON file mapping labels to human-readable names')

    arguments = parser.parse_args()

    model = load_trained_model(arguments.model_file)
    probabilities, class_indices = make_prediction(arguments.image_file, model, arguments.top_k)

    with open(arguments.label_map, 'r') as json_file:
        labels_map = json.load(json_file)
        readable_class_labels = [labels_map[str(index)] for index in class_indices]

    print('Predicted Classes:', readable_class_labels)
    print('Associated Probabilities:', probabilities)

if __name__ == '__main__':
    main()
