import argparse
from pathlib import Path

from video_processor import VideoProcessor
from video_cropper import VideoCropper
from video_detector import VideoDetector
from video_configuration import DEFAULT_SETTINGS, Config
from video_annotator import VideoAnnotator

def parse_arguments():
    parser = argparse.ArgumentParser(description="Video Processing Tool")
    parser.add_argument('-f', dest='file_path', action='store')
    parser.add_argument('-d', dest='dir_path', action='store')
    
    parser.add_argument("--detector", choices=['mpipeseg', 'mpipelmark', 'yolo'], default=DEFAULT_SETTINGS['detector'])
    parser.add_argument("--multiple", action="store_true", default=DEFAULT_SETTINGS['multiple'])

    parser.add_argument("--body", choices=['portrait', 'waist', 'legs', 'full_body'], default=DEFAULT_SETTINGS['body'])

    parser.add_argument("--process", nargs='*', choices=['zoom', 'stable', 'no_outliers', 'pose_anchor'], default=DEFAULT_SETTINGS['process'])
    parser.add_argument("--strategy", nargs='+', choices=['basic', 'adaptive_movement', 'adaptive_zoom'], default=DEFAULT_SETTINGS['strategy'])
    parser.add_argument("--movement", choices=['free', 'horizontal', 'vertical', 'fullbox'], default=DEFAULT_SETTINGS['movement'])
    parser.add_argument("--size", type=parse_size, default=DEFAULT_SETTINGS['size'], help="Set the size of the output in 'W*H' format, e.g., '9*16'.")
    parser.add_argument("--annotate", nargs='+', choices=['landmarks', 'bbox', 'fbox'], default=DEFAULT_SETTINGS['annotate'])
    
    args = parser.parse_args()

    if not args.file_path and not args.dir_path:
        parser.error('Either a file path (-f) or a directory path (-d) must be provided.')
    return args

def parse_size(s):
    if s == "minimal":
        return s
    try:
        width, height = map(int, s.split('*'))
        return (width, height)
    except ValueError:
        raise argparse.ArgumentTypeError("Size must be in 'W*H' format, e.g., '9*16'.")


def main():
    args = parse_arguments()

    if (args.file_path):
        process_video(args.file_path, args)
    else:
        all_files = Path(args.dir_path).glob('*')  
        for file_path in all_files:
            if file_path.is_file(): 
                process_video(str(file_path), args)

def process_video(file_path, args):

    Config.initialize(args, file_path)

    detector = VideoDetector(Config)
    detector.process()

    processor = VideoProcessor(Config)
    processor.process()

    annotator = VideoAnnotator(Config)
    annotator.process()

    cropper = VideoCropper(Config)
    cropper.process()


if __name__ == "__main__":
    main()
