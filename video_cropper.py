import cv2
import json

class VideoCropper:
    def __init__(self, config):
        self.config = config
        self.stats = None
        self.fbox = None

    def crop(self):
        cap = cv2.VideoCapture(self.config.src)
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')

        out = cv2.VideoWriter(self.config.get_video_path(include_full_config=True), 
        fourcc, 
        self.stats['fps'], 
        (int(self.stats['out_width']), 
         int(self.stats['out_height'])))

        frame_index = 0

        if not cap.isOpened():
            print(f"Error opening video file: {self.config.src}")
            return
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                point = self.fbox[frame_index]
                cropped_image = frame[int(point['minY']):int(point['maxY']), int(point['minX']):int(point['maxX'])]
                out.write(cropped_image)
            else:
                break
            frame_index += 1
        cap.release()
        out.release()

    def crop_with_zoom(self):
        cap = cv2.VideoCapture(self.config.src)
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')

        out = cv2.VideoWriter(self.config.get_video_path(include_full_config=True), 
        fourcc, 
        self.stats['fps'], 
        (int(self.stats['out_width']), 
         int(self.stats['out_height'])))

        frame_index = 0

        if not cap.isOpened():
            print(f"Error opening video file: {self.config.src}")
            return
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                point = self.fbox[frame_index]
                cropped_image = frame[int(point['minY']):int(point['maxY']), int(point['minX']):int(point['maxX'])]
                resized_image = cv2.resize(cropped_image, (self.stats['out_width'], self.stats['out_height']), interpolation = cv2.INTER_AREA)
                out.write(resized_image)
            else:
                break
            frame_index += 1
        cap.release()
        out.release()

    def process(self):
        with open(self.config.get_log_path(include_stats=True), "r") as stat_file:
            self.stats = json.load(stat_file)
        
        with open(self.config.get_log_path(include_fbox=True), "r") as fbox_file:
            self.fbox = json.load(fbox_file)

        if 'zoom' in self.config.process:
            self.crop_with_zoom()
        else:
            self.crop()