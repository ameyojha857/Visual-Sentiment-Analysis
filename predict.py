import numpy as np
import tensorflow as tf
from keras.preprocessing import image

# List of model paths
model_paths = ['vgg16.tflite', 'vgg19.tflite', 'resnet101.tflite', 'resnet152.tflite', 'inceptionv3.tflite']

# Placeholder labels in the order they appear in the model output
emotion_labels = ["Emotion_0", "Emotion_1", "Emotion_2", "Emotion_3", "Emotion_4", "Emotion_5", "Emotion_6", "Emotion_7"]

# Dictionary to store predicted scores for each emotion
emotion_scores = {emotion: [] for emotion in emotion_labels}

# Loop through each model
for model_path in model_paths:
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Load and preprocess the image
    img_path = r'C:\Users\Ameyo\OneDrive\Desktop\Visual-Sentiment-Analysis-in-Python-master\downloaded_images\fear\image (1).jpg'  # Replace with the path to your image
    img = image.load_img(img_path, target_size=(96, 96))
    img = np.asarray(img)
    img = img.astype(np.float32)  # Ensure the image is in the correct data type
    img = (img - 127.5) / 127.5  # Normalize the image to the range [-1, 1]
    img = np.expand_dims(img, axis=0)

    # Set the input tensor to the loaded image
    interpreter.set_tensor(input_details[0]['index'], img)

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    output = interpreter.get_tensor(output_details[0]['index'])[0]

    # Store the output scores in the dictionary
    for i, score in enumerate(output):
        emotion_scores[emotion_labels[i]].append(score)

# Print scores with labels
for emotion, scores in emotion_scores.items():
    print(f"Emotion: {emotion}")
    for model_idx, score in enumerate(scores):
        model_name = model_paths[model_idx]
        print(f"  Model {model_name}: {score}")
