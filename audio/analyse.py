from moviepy.editor import VideoFileClip, AudioFileClip
from torch.cuda import current_stream
from torch.utils.data import Dataset, DataLoader
import librosa
import logging
import numpy as np
import os
import sys
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


logging.getLogger("moviepy").setLevel(logging.ERROR)
classes = [
    "gamingtype",
    "minimalisttype",
    "mrbeasttype",
    "techreviewtype",
    "testvideos",
    "vlogtype",
]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_audio_features2(file_path, mfcc=True, chroma=True, mel=True):
    # Load audio file
    waveform, sample_rate = torchaudio.load(file_path)

    # Convert waveform to mono if it's stereo
    if waveform.size(0) > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Extract features
    result = np.array([])
    if mfcc:
        mfccs = torchaudio.transforms.MFCC(sample_rate)(waveform)
        mfccs = (
            torch.mean(mfccs, dim=2).squeeze().numpy()
        )  # Collapse the time dimension
        result = np.hstack((result, mfccs))
    if chroma:
        chroma = librosa.feature.chroma_stft(y=waveform.numpy()[0], sr=sample_rate)
        chroma = np.mean(chroma, axis=1)
        result = np.hstack((result, chroma))
    if mel:
        mel = torchaudio.transforms.MelSpectrogram(sample_rate)(waveform)
        mel = torch.mean(mel, dim=2).squeeze().numpy()  # Collapse the time dimension
        result = np.hstack((result, mel))
    return result


class AudioModel(nn.Module):
    def __init__(self, input_channels, output_size):
        super(AudioModel, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(
            64 * 45, 128
        )  # Adjusted output size based on input dimensions
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout1(x)
        x = x.view(-1, 64 * 45)  # Flatten before passing to fully connected layer
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)


output_size = len(classes)
in_channels = 1

model_path = "./audio/models/model.pth"
model = AudioModel(in_channels, output_size).to(
    device
)  # Make sure to create an instance of your model before loading the state_dict
model.load_state_dict(torch.load(model_path))
model.eval()


def classify_segment(segment_file):
    features = extract_audio_features2(segment_file)
    features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        outputs = model(features)

    probabilities = torch.softmax(outputs, dim=1).squeeze().numpy()
    prediction = np.argmax(probabilities)
    class_probabilities = {classes[i]: probabilities[i] for i in range(len(classes))}

    return prediction, class_probabilities


def classify_all_segments(segment_files, segment_duration=10):
    class_probabilities_sum = {label: 0.0 for label in classes}
    total_segments = len(segment_files)
    all_predictions = []
    current_segment = 0
    for segment_file in segment_files:
        prediction, class_probabilities = classify_segment(segment_file)
        prediction = classes[prediction]
        timestamp = {"time": current_segment, "prediction": prediction}

        if current_segment == 0:
            all_predictions.append(timestamp)

        elif all_predictions[-1]["prediction"]!=timestamp["prediction"]:
            all_predictions.append(timestamp)

        current_segment += segment_duration

        for label, probability in class_probabilities.items():
            class_probabilities_sum[label] += probability

    average_probabilities = {
        label: class_probabilities_sum[label] / total_segments for label in classes
    }
    return all_predictions, average_probabilities


def get_segment_files(directory):
    segment_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith("segment_") and file.endswith(".mp3"):
                segment_files.append(os.path.join(root, file))
    return segment_files


def video_to_audio(video_file, audio_file):
    video = VideoFileClip(video_file)
    audio = video.audio
    audio.write_audiofile(audio_file)


def segment_audio(audio_file, output_dir, segment_duration=10):
    audio = AudioFileClip(audio_file)
    duration = audio.duration
    segments = int(duration // segment_duration)

    for i in range(segments):
        start_time = i * segment_duration
        end_time = min((i + 1) * segment_duration, duration)
        segment = audio.subclip(start_time, end_time)
        segment_file = os.path.join(output_dir, f"segment_{i+1}.mp3")
        segment.write_audiofile(segment_file)

    # Handle the last segment
    if duration % segment_duration != 0:
        start_time = segments * segment_duration
        segment = audio.subclip(start_time, duration)
        segment_file = os.path.join(output_dir, f"segment_{segments+1}.mp3")
        segment.write_audiofile(segment_file)


def analyse_audio(video_file):
    segment_duration = 10
    output_dir = "./audio/output"

    os.system("rm -rf " + output_dir + "/*")
    audio_file = os.path.join(output_dir, "audio.mp3")
    video_to_audio(video_file, audio_file)

    segment_audio(audio_file, output_dir, segment_duration)

    os.system("rm -rf ./audio/output/audio.mp3")
    segment_files = get_segment_files(output_dir)

    all_predictions, average_predictions = classify_all_segments(segment_files, segment_duration)
    print("Average predictions:", average_predictions)
    os.system("rm -rf ./audio/output/*")

    result = {"all_predictions":all_predictions, "average_predictions": average_predictions}

    return result

