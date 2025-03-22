from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os
import re

# Initialize the BLIP processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Function to generate caption for an image
def generate_caption(image_path):
    """Generate caption for an image using BLIP model."""
    raw_image = Image.open(image_path).convert("RGB")
    inputs = processor(raw_image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# Function to clean captions to make them safe for filenames
def clean_caption_for_filename(caption):
    """Clean the caption to make it a valid filename."""
    caption = caption.lower()  # Convert to lowercase
    caption = re.sub(r'[^\w\s-]', '', caption)  # Remove non-alphanumeric characters
    caption = re.sub(r'[-\s]+', '_', caption)  # Replace spaces and hyphens with underscores
    caption = caption.strip()  # Remove any leading/trailing whitespace
    return caption

# Function to rename frames based on captions
def rename_frames_based_on_caption(frames_root_dir):
    """Rename frames in subfolders based on their generated captions."""
    for subfolder in os.listdir(frames_root_dir):
        subfolder_path = os.path.join(frames_root_dir, subfolder)
        
        # Skip if it's not a directory
        if not os.path.isdir(subfolder_path):
            continue
        
        print(f"Processing frames in subfolder: {subfolder}")

        # List all the files in the subfolder (only .jpg files)
        frames = [f for f in os.listdir(subfolder_path) if f.endswith(".jpg")]
        frames.sort()  # Sort the frames if they are not in order

        for frame in frames:
            frame_path = os.path.join(subfolder_path, frame)
            caption = generate_caption(frame_path)  # Generate a caption for the frame
            cleaned_caption = clean_caption_for_filename(caption)  # Clean the caption for filenames

            # Create the new filename (using cleaned caption)
            new_frame_name = f"{cleaned_caption}.jpg"  # Use the cleaned caption as the new filename
            new_frame_path = os.path.join(subfolder_path, new_frame_name)

            # Rename the file
            os.rename(frame_path, new_frame_path)
            print(f"Renamed {frame} to {new_frame_name}")

# Path to your frames directory
frames_directory = "/home/lab/workspace/Jordan/VITA/frames"
rename_frames_based_on_caption(frames_directory)
