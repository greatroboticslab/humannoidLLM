from moviepy.editor import VideoFileClip
import os

def extract_audio_from_videos(video_folder, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each video in the folder
    for filename in os.listdir(video_folder):
        if filename.endswith((".mp4", ".mkv", ".avi", ".mov")):  # Add other video formats if needed
            video_path = os.path.join(video_folder, filename)
            output_audio_path = os.path.join(output_folder, filename.rsplit(".", 1)[0] + ".wav")

            # Extract audio
            video = VideoFileClip(video_path)
            audio = video.audio
            audio.write_audiofile(output_audio_path)

            print(f"Extracted audio from {filename} to {output_audio_path}")

# Example usage
video_folder = "/home/lab/Videos"  # Folder containing multiple videos
output_folder = "/home/lab/workspace/Jordan/VITA/audio"  # Folder to save extracted audio files

extract_audio_from_videos(video_folder, output_folder)