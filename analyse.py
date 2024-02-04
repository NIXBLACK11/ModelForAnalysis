from moviepy.video.io.VideoFileClip import VideoFileClip
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np
import librosa
import os
from tqdm import tqdm
from PIL import Image

from concurrent.futures import ProcessPoolExecutor
from functools import partial

tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow warnings

model_image_path = 'video_analysis_vgg16_adamax.h5'
model_image = load_model(model_image_path)
model_audio_path = 'audio_analysis_multi_genre.h5'
model_audio = load_model(model_audio_path)

genres = ['MrBeastType', 'VlogType', 'TechReviewType', 'GamingType', 'MinimalistType']

def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def predict_genre(img):
    img_array = preprocess_image(img)
    prediction = model_image.predict(img_array)
    predicted_genre = genres[np.argmax(prediction)]
    return predicted_genre

def extract_features(audio_data, sr):
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
    return mfccs

def process_audio_segment(audio_data, model_audio, genres):
    sr = 44100  # Set your desired sample rate
    audio_features = extract_features(audio_data, sr)

    expected_shape = (216, 13)
    if audio_features.shape[1] < expected_shape[0]:
        audio_features = np.pad(audio_features, ((0, 0), (0, expected_shape[0] - audio_features.shape[1])))
    elif audio_features.shape[1] > expected_shape[0]:
        audio_features = audio_features[:, :expected_shape[0]]

    audio_features = audio_features.transpose(1, 0)
    audio_features = audio_features.reshape(1, audio_features.shape[0], audio_features.shape[1])

    audio_features = audio_features[:, :expected_shape[0]]
    prediction_audio = model_audio.predict(audio_features)
    predicted_genre_audio = genres[np.argmax(prediction_audio)]

    return predicted_genre_audio

def analyseScreenshot(video_path):
    video_clip = VideoFileClip(video_path)
    genre_counts = {genre: 0 for genre in genres}

    for timestamp in tqdm(range(0, int(video_clip.duration), 10), desc="Processing Screenshots"):
        screenshot = video_clip.get_frame(timestamp)

        pil_image = Image.fromarray(screenshot)

        predicted_genre = predict_genre(pil_image)

        genre_counts[predicted_genre] += 1

    video_clip.close()
    return {'genre_counts': genre_counts}

def analyseAudio(video_path):
    video_clip = VideoFileClip(video_path)
    audio = video_clip.audio
    chunk_duration = 360  # seconds

    genre_counts_audio = {genre: 0 for genre in genres}

    for start_time in tqdm(range(0, int(video_clip.duration), chunk_duration), desc="Processing Audio"):
        end_time = min(start_time + chunk_duration, int(video_clip.duration))
        audio_chunk = audio.subclip(start_time, end_time)

        audio_data, _ = librosa.load(audio_chunk.filename, sr=None)

        predicted_genre_audio = process_audio_segment(audio_data, model_audio, genres)

        genre_counts_audio[predicted_genre_audio] += 1

    video_clip.close()
    return {'genre_counts_audio': genre_counts_audio}

def analyseVideo(video_path):
    results_screenshot = analyseScreenshot(video_path)
    results_audio = analyseAudio(video_path)

    # Combine the results from both analyses
    combined_results = {
        'genre_counts': results_screenshot['genre_counts'],
        'genre_counts_audio': results_audio['genre_counts_audio']
    }

    return combined_results
