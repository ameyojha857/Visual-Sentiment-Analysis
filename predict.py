import numpy as np
import tensorflow as tf
from keras.preprocessing import image

model_paths = ['vgg16.tflite', 'vgg19.tflite', 'resnet101.tflite', 'resnet152.tflite', 'inceptionv3.tflite']
emotion_labels = ["Anger", "Content", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
emotion_scores = {emotion: [] for emotion in emotion_labels}

for model_path in model_paths:
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    img_path = r'C:\Users\Ameyo\OneDrive\Desktop\Visual-Sentiment-Analysis-in-Python-master\downloaded_images\fear\image (1).jpg'  
    img = image.load_img(img_path, target_size=(96, 96))
    img = np.asarray(img)
    img = img.astype(np.float32)  
    img = (img - 127.5) / 127.5 
    img = np.expand_dims(img, axis=0)
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    
    for i, score in enumerate(output):
        emotion_scores[emotion_labels[i]].append(score)
        for emotion, scores in emotion_scores.items():
    print(f"Emotion: {emotion}")
    for model_idx, score in enumerate(scores):
        model_name = model_paths[model_idx]
        print(f"  Model {model_name}: {score}")
