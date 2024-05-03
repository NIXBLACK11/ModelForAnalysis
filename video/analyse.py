import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image


model_path = './video/models/video_analysis_vgg16_adamax.h5'
model = load_model(model_path)

classes = ['gamingtype', 'minimalisttype', 'mrbeasttype', 'techreviewtype', 'vlogtype']

def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array


def segment_video(video_path, output_folder, chunk_size):
    chunk_files = []
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video_capture = cv2.VideoCapture(video_path)
    frame_count = 0
    frame_rate = video_capture.get(cv2.CAP_PROP_FPS)
    frame_interval = int(frame_rate * chunk_size)  # Capture a frame every 10 seconds

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        frame_count += 1
        # Capture a frame every 10 seconds
        if frame_count % frame_interval == 0:
            frame_name = f"frame_{frame_count}.png"
            frame_path = os.path.join(output_folder, frame_name)
            cv2.imwrite(frame_path, frame)
            chunk_files.append(frame_path)
            print(f"Saved {frame_name}")

    # Release the video capture object
    video_capture.release()
    return chunk_files


def analyse_video(video_path, chunk_size=10, output_folder='./video/output'):
    class_probabilities_sum = {label: 0.0 for label in classes}
    chunk_files = segment_video(video_path, output_folder, chunk_size)
    total_segments = len(chunk_files)
    current_segment = 0
    all_predictions = []
    for chunk_file in chunk_files:
        pil_image = Image.open(chunk_file)
        pil_image = preprocess_image(pil_image)
        class_probabilities = model.predict(pil_image)

        prediction = classes[np.argmax(class_probabilities)]
        timestamp = {"time":current_segment, "prediction": prediction}

        if current_segment==0:
            all_predictions.append(timestamp)

        elif all_predictions[-1]["prediction"]!=timestamp["prediction"]:
            all_predictions.append(timestamp)

        current_segment += chunk_size

        for i, label in enumerate(classes):
            class_probabilities_sum[label] += class_probabilities[0][i]
    average_probabilities = {label.lower(): class_probabilities_sum[label] / total_segments for label in classes}

    result = {"average_probabilities": average_probabilities, "all_predictions": all_predictions}
    os.system("rm ./video/output/*")
    return result
