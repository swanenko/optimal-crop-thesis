import cv2
import numpy as np
import argparse

# def stack_side_by_side(videos):
#     # Function to stack videos side by side, centering them vertically with white backgrounds
#     max_height = max(video.shape[0] for video in videos)
#     total_width = sum(video.shape[1] for video in videos)
#     output_video = 255 * np.ones((max_height, total_width, 3), dtype=np.uint8)  # Fill with white

#     current_x = 0
#     for video in videos:
#         height, width, _ = video.shape
#         start_y = (max_height - height) // 2  # Calculate starting y position for centering
#         output_video[start_y:start_y + height, current_x:current_x + width] = video
#         current_x += width
    
#     return output_video

def stack_on_top(videos):
    # Function to stack videos on top of each other, centering them horizontally with white backgrounds
    max_width = max(video.shape[1] for video in videos)  # Find the maximum width among all videos
    total_height = sum(video.shape[0] for video in videos)  # Sum of all heights for stacking vertically
    output_video = 255 * np.ones((total_height, max_width, 3), dtype=np.uint8)  # Fill with white

    current_y = 0
    for video in videos:
        height, width, _ = video.shape
        start_x = 0  # Calculate starting x position for centering
        output_video[current_y:current_y + height, start_x:start_x + width] = video
        current_y += height  # Move the starting point for the next video down by the height of the current video
    
    return output_video


def picture_in_picture(videos):
    # Function for picture in picture effect, using the original size of the smaller video
    base_video = videos[0]
    overlay_video = videos[1]

    # Ensure the overlay video is smaller; otherwise, take a portion to fit the bottom-right corner
    overlay_height, overlay_width = overlay_video.shape[0], overlay_video.shape[1]
    base_height, base_width = base_video.shape[0], base_video.shape[1]

    # Position the smaller video in the bottom-right corner of the larger video
    start_y = base_height - overlay_height
    start_x = base_width - overlay_width
    base_video[start_y:start_y + overlay_height, start_x:start_x + overlay_width] = overlay_video

    return base_video

def process_videos(video_paths, mode, output_path):
    caps = [cv2.VideoCapture(path) for path in video_paths]
    frames_exist = [cap.isOpened() for cap in caps]

    if not all(frames_exist):
        print("Error opening one or more video files.")
        return

    fps = max(int(cap.get(cv2.CAP_PROP_FPS)) for cap in caps)  # Use the highest FPS for smoothest playback
    if mode == 'side_by_side':
        frame_height = sum(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) for cap in caps)
        frame_width = max(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) for cap in caps)
    else:  # picture_in_picture mode
        frame_width, frame_height = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH)), int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    while all(frames_exist):
        frames = []
        for cap in caps:
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            else:
                frames_exist = [False]
                break

        if not all(frames_exist):
            break

        if mode == 'side_by_side':
            output_frame = stack_on_top(frames)
        elif mode == 'picture_in_picture':
            output_frame = picture_in_picture(frames)

        out.write(output_frame)

    for cap in caps:
        cap.release()
    out.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Video Processing Tool')
    parser.add_argument('videos', nargs='+', help='Paths to video files')
    parser.add_argument('--mode', choices=['side_by_side', 'picture_in_picture'], required=True, help='Mode of processing videos')
    parser.add_argument('--output', required=True, help='Output video file path')
    
    args = parser.parse_args()

    if len(args.videos) < 2 or len(args.videos) > 4:
        print("Error: Please provide between 2 to 4 video paths.")
        return

    process_videos(args.videos, args.mode, args.output)

if __name__ == '__main__':
    main()
