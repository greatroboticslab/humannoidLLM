from concurrent.futures import ProcessPoolExecutor
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os
import torch
import multiprocessing

# Initialize CUDA
def generate_caption(image_path):
    """
    Generate a caption for an image using a pre-trained BLIP model.
    Initialize the model and processor in each process to avoid CUDA issues.
    """
    try:
        # Initialize the processor and model inside the worker function
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-small")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-small").to(device)

        raw_image = Image.open(image_path).convert("RGB")
        inputs = processor(raw_image, return_tensors="pt").to(device)
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        
        # Clear GPU memory after processing
        torch.cuda.empty_cache()
        
        return image_path, caption
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return image_path, None


def process_images(image_folder):
    """
    Process all images in the folder (including subfolders) and generate captions for them.
    Process images sequentially to reduce memory usage.
    """
    image_paths = []

    # Traverse the folder and its subfolders to collect all .jpg image paths
    for root, _, files in os.walk(image_folder):
        for file in files:
            if file.endswith(".jpg"):
                image_paths.append(os.path.join(root, file))

    results = []
    for image_path in image_paths:
        result = generate_caption(image_path)
        results.append(result)

    return results


if __name__ == "__main__":
    # Set the multiprocessing start method to 'spawn'
    multiprocessing.set_start_method('spawn', force=True)

    # Example usage
    image_folder = "/home/lab/workspace/Jordan/VITA/frames"  # Folder containing all frames, including subfolders
    captions = process_images(image_folder)

    # Now, `captions` will contain tuples of image paths and their captions
    for image_path, caption in captions:
        if caption:
            print(f"Image: {image_path}, Caption: {caption}")
        else:
            print(f"Image: {image_path}, Caption: Error generating caption")
