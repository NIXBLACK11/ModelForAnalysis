import os
import sys
from moviepy.editor import VideoFileClip, AudioFileClip
import logging
logging.getLogger("moviepy").setLevel(logging.ERROR)


SEGMENT_DURATION = 10
def video_to_audio(video_file, audio_file):
    video = VideoFileClip(video_file)
    audio = video.audio
    audio.write_audiofile(audio_file)

def segment_audio(audio_file, output_dir, segment_duration=SEGMENT_DURATION):
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


def segment_video(video_file):
    output_dir = "output"
    audio_file = os.path.join(output_dir, "audio.mp3")
    video_to_audio(video_file, audio_file)
    print("[*] Converted video to audio...")
    segment_audio(audio_file, output_dir)
    print("[*] Audio Segmentation finished...")

