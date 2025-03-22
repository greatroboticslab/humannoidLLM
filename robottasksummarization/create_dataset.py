import os
import ffmpeg
import json
from pathlib import Path
from PIL import Image
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
from transformers import BlipProcessor, BlipForConditionalGeneration

# Initialize BLIP captioning model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(image_path):
    """
    Generate a caption for an image using a pre-trained BLIP model.
    """
    # Open the image
    raw_image = Image.open(image_path).convert("RGB")
    
    # Preprocess the image and feed it to the model
    inputs = processor(raw_image, return_tensors="pt")
    
    # Generate the caption
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# Define the function `extract_frames_and_audio` here (before using it)
def extract_frames_and_audio(video_path, frames_folder, audio_folder, frame_duration=1):
    """
    Extract frames and audio from the video file.
    """
    video_name = Path(video_path).stem
    video_frames_folder = os.path.join(frames_folder, video_name)
    video_audio_folder = os.path.join(audio_folder, video_name)

    os.makedirs(video_frames_folder, exist_ok=True)
    os.makedirs(video_audio_folder, exist_ok=True)

    # Extract frames
    ffmpeg.input(video_path).output(f"{video_frames_folder}/frame_%06d.jpg").run()

    # Extract audio
    ffmpeg.input(video_path).output(f"{video_audio_folder}/audio.%06d.wav", f="segment", segment_time=frame_duration).run()

    print(f"Frames and audio extracted for {video_name}")

def create_dataset(frames_folder, audio_folder, output_json):
    """
    Create a dataset from extracted frames and audio, with captions.
    """
    dataset = []
    
    for video_folder in sorted(os.listdir(frames_folder)):
        video_folder_path = os.path.join(frames_folder, video_folder)
        
        if not os.path.isdir(video_folder_path):
            continue
        
        audio_folder_path = os.path.join(audio_folder, video_folder)
        
        frame_data_list = []
        
        for frame_name in sorted(os.listdir(video_folder_path)):
            if not frame_name.endswith(".jpg"):
                continue

            frame_path = os.path.join(video_folder_path, frame_name)
            audio_name = frame_name.replace(".jpg", ".wav")
            audio_path = os.path.join(audio_folder_path, audio_name)
            
            if not os.path.exists(audio_path):
                print(f" Audio file not found for {frame_name}, skipping...")
                continue

            frame_data_list.append((frame_path, audio_path, frame_name))
        
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_frame, frame_data) for frame_data in frame_data_list]
            for future in as_completed(futures):
                entry = future.result()
                dataset.append(entry)
    
    # Save dataset to JSON
    with open(output_json, "w") as f:
        json.dump(dataset, f, indent=4)

    print(f"✅ Dataset saved to {output_json}")


# Example usage
videos_folder = "/home/lab/Videos"
frames_folder = "/home/lab/workspace/Jordan/VITA/frames"
audio_folder = "/home/lab/workspace/Jordan/VITA/audio"
output_json = "/home/lab/workspace/Jordan/VITA/dataset/dataset.json"

# Extract frames and audio for all videos
for video in sorted(os.listdir(videos_folder)):
    video_path = os.path.join(videos_folder, video)
    if video.endswith(".mp4"):
        extract_frames_and_audio(video_path, frames_folder, audio_folder)

# Create dataset from extracted frames, audio, and captions
create_dataset(frames_folder, audio_folder, output_json)
