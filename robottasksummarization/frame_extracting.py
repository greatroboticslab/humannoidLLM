import os
from pathlib import Path
import ffmpeg
from concurrent.futures import ProcessPoolExecutor, as_completed

def extract_frames(video_path, frames_folder, frame_duration=1):
    """
    Extract frames from a video file.
    """
    video_name = Path(video_path).stem
    video_frames_folder = os.path.join(frames_folder, video_name)

    os.makedirs(video_frames_folder, exist_ok=True)

    # Extract frames using ffmpeg
    ffmpeg.input(video_path).output(f"{video_frames_folder}/frame_%06d.jpg", vsync=0, framerate=1).run()

    print(f"Frames extracted for {video_name}")

def process_video_frames(video_path, frames_folder, frame_duration=1):
    """
    Process each video by extracting frames.
    This function will handle one video at a time.
    """
    try:
        extract_frames(video_path, frames_folder, frame_duration)
    except Exception as e:
        print(f"Error processing {video_path}: {e}")

def process_multiple_videos(videos_folder, frames_folder, frame_duration=1):
    """
    Process multiple videos and extract frames.
    """
    video_paths = [os.path.join(videos_folder, video) for video in os.listdir(videos_folder) if video.endswith(".mp4")]
    
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_video_frames, video_path, frames_folder, frame_duration) for video_path in video_paths]
        
        for future in as_completed(futures):
            future.result()

    print("✅ Frame extraction completed for all videos.")

# Example usage
videos_folder = "/path/to/your/videos"
frames_folder = "/path/to/save/frames"

# Process multiple videos and extract frames
process_multiple_videos(videos_folder, frames_folder)
