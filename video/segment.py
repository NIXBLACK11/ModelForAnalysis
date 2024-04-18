import cv2
import sys
import os
import concurrent.futures


def capture_frames(video_path, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    file_name = os.path.splitext(os.path.basename(video_path))[0]
    # Open the video file
    video_capture = cv2.VideoCapture(video_path)
    frame_count = 0
    frame_rate = video_capture.get(cv2.CAP_PROP_FPS)
    frame_interval = int(frame_rate * 10)  # Capture a frame every 10 seconds

    while True:
        ret, frame = video_capture.read()

        if not ret:
            break

        frame_count += 1

        # Capture a frame every 10 seconds
        if frame_count % frame_interval == 0:
            frame_name = f"frame_{file_name}_{frame_count // frame_interval}.png"
            frame_path = os.path.join(output_folder, frame_name)
            cv2.imwrite(frame_path, frame)
            print(f"Saved {frame_name}")

    # Release the video capture object
    video_capture.release()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python script.py input_folder")
    input_folder = sys.argv[-1]
    output_folder = input_folder.lower()
    videos = []
    for video_file in os.listdir(input_folder):
        video_path = os.path.join(input_folder, video_file)
        if os.path.isfile(video_path):
            videos.append(video_path)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(capture_frames, videos, [output_folder] * len(videos))

