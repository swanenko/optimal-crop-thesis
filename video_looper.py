import cv2
import argparse

def loop_video(input_path, output_path, loop_duration=60):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = frame_count / fps

    loop_count = int(loop_duration // video_duration)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Setting up the writer with the same codec, fps and size as the original video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change 'mp4v' to 'XVID' if you want to save as AVI
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    for _ in range(loop_count):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Rewind to the start of the video
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

    cap.release()
    out.release()
    print(f"Video looped successfully to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Loop a video to a specified duration and save the output.")
    parser.add_argument("input", help="Path to the input video file.")
    parser.add_argument("output", help="Path to the output video file.")
    
    args = parser.parse_args()
    loop_video(args.input, args.output)

if __name__ == "__main__":
    main()
