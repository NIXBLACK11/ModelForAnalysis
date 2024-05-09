from moviepy.video.io.VideoFileClip import VideoFileClip
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from tqdm import tqdm
from PIL import Image


model_path = './largeFiles/video_analysis_vgg16_adamax.h5'
model = load_model(model_path)

genres = ['MrBeastType', 'VlogType', 'TechReviewType', 'GamingType', 'MinimalistType']

def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def predict_genre(img):
    img_array = preprocess_image(img)
    prediction = model.predict(img_array)
    predicted_genre = genres[np.argmax(prediction)]
    return predicted_genre

def analyze_video(video_path):
    video_clip = VideoFileClip(video_path)
    chunk_size = 10
    class_probabilities_sum = {label: 0.0 for label in genres}
    total_segments = 0

    for i, chunk in enumerate(tqdm(video_clip.iter_frames(fps=video_clip.fps), desc="Processing Chunks")):
        timestamp = i * chunk_size
        pil_image = Image.fromarray(chunk)

        predicted_genre = predict_genre(pil_image)

        class_probabilities = model.predict(preprocess_image(pil_image))[0]
        class_probabilities_sum[predicted_genre] += max(class_probabilities)
        total_segments += 1

    video_clip.close()

    average_probabilities = {label.lower(): class_probabilities_sum[label] / total_segments for label in genres}
    return average_probabilities


video_path = "/home/rocky/test/audio/videoplayback.mp4"
videoGenre = "MrBeastType"
average_probabilities = analyze_video(video_path,videoGenre)
print(average_probabilities)
