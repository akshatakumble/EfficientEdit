import cv2
import os


def extract_frame_at_fraction(video_path, output_path, fraction):
    """
    Extracts a frame from a video at a specific fractional point.

    Args:
        video_path (str): Path to the source MP4 video.
        output_path (str): Path (including filename) to save the image.
        fraction (float): A number between 0.0 and 1.0 (e.g., 0.5 for middle).
    """

    # 1. Validate inputs
    if not (0 <= fraction <= 1):
        raise ValueError("Fraction must be between 0.0 and 1.0")

    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    # 2. Capture the video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    try:
        # 3. specific property retrieval
        # Get total number of frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate the index of the frame we want
        target_frame_index = int(total_frames * fraction)

        # Handle edge case: if fraction is 1.0, grab the last valid frame index
        if target_frame_index >= total_frames:
            target_frame_index = total_frames - 1

        # 4. Seek to the specific frame
        #
        print(f"Seeking to frame {target_frame_index} of {total_frames}...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_index)

        # 5. Read the frame
        success, frame = cap.read()

        if success:
            # 6. Save the result
            cv2.imwrite(output_path, frame)
            print(f"Success! Frame saved to: {output_path}")
        else:
            print("Error: Could not read the frame at the specified position.")

    finally:
        # 7. Release resources
        cap.release()


# --- Example Usage ---
if __name__ == "__main__":
    # Create a dummy path examples
    input_video = r'D:\downloads\GenPropRepo\val_data\masks\4O8mr9iSBdw.mp4'
    output_image = "mask_frame.jpg"

    # Example: Get the frame exactly halfway through (0.5)
    # Note: Replace 'example_video.mp4' with your actual file path to test
    if os.path.exists(input_video):
        extract_frame_at_fraction(input_video, output_image, 0.5)
    else:
        print(f"Please place a video file named '{input_video}' in this directory to test.")