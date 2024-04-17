import json
import os
import numpy as np
import matplotlib.pyplot as plt

from video_detector import VideoDetector

body_configurations = {
    "portrait": range(0, 12),
    "waist": range(0, 24),
    "legs": range(25, 32)
}

class VideoProcessor:
    def __init__(self, config):
        self.config = config
        self.bbox = None
        self.dimensions = None
        self.fbox = []

    def analyze_movement_ranges(self, bounding_boxes):
        min_x_values = [bbox['minX'] for bbox in bounding_boxes]
        min_y_values = [bbox['minY'] for bbox in bounding_boxes]

        range_min_x = max(min_x_values) - min(min_x_values)
        range_min_y = max(min_y_values) - min(min_y_values)

        normalized_range_x = range_min_x / self.config.metadata['frame_width']
        normalized_range_y = range_min_y / self.config.metadata['frame_height']

        print(f"Range of minX values: {normalized_range_x}")
        print(f"Range of minY values: {normalized_range_y}")

        if normalized_range_x > normalized_range_y:
            print("Primary movement is horizontal.")
        else:
            print("Primary movement is vertical.")


    def free(self):

        for bbox in self.bbox:

            bbox_width = bbox['maxX'] - bbox['minX']
            bbox_height = bbox['maxY'] - bbox['minY']
          
            center_x = bbox['minX'] + (bbox_width//2)
            center_y = bbox['minY'] + (bbox_height//2)

            frame_min_x = center_x - (self.dimensions[0] // 2)
            frame_min_y = center_y - (self.dimensions[1] // 2)

            frame_max_x = frame_min_x + self.dimensions[0] 
            frame_max_y = frame_min_y + self.dimensions[1]

            self.fbox.append({
                'minX': frame_min_x,
                'minY': frame_min_y,
                'maxX': frame_max_x,
                'maxY': frame_max_y,
            })

    def horizontal(self, bounding_boxes):
        min_y_values = [bbox['minY'] for bbox in bounding_boxes]
        max_y_values = [bbox['maxY'] for bbox in bounding_boxes]

        self.dimensions[1] = max(max_y_values) - min(min_y_values)
        
        range_min_y = max(min_y_values) - min(min_y_values)

        normalized_range_x = range_min_x / self.config.metadata['frame_width']
        normalized_range_y = range_min_y / self.config.metadata['frame_height']


    def vertical():
        pass

    def fullbox():
        pass

    def get_video_dimensions(self):
        widths = []
        heights = []
        for point in self.bbox:
            if point:
                widths.append(point['maxX']-point['minX'])
                heights.append(point['maxY']-point['minY'])

        self.dimensions = (max(widths), max(heights))

    def process_body_dimensions(self, body_mode):
        if (os.path.exists(self.config.get_log_path(custom_name=str(body_mode)))):
            with open(self.config.get_log_path(custom_name=str(body_mode)), 'r') as file:
                self.bbox = json.load(file)
        
        try:
            with open(self.config.get_log_path(custom_name="landmarks"), 'r') as file:
                landmarks = json.load(file)
        except:
            print("File with landmarks not found. Creating one...")
            self.config.detector = 'mpipelmark'
            detector = VideoDetector(self.config)
            detector.process()
            with open(self.config.get_log_path(custom_name="landmarks"), 'r') as file:
                landmarks = json.load(file)

        result = []
        body_range = body_configurations.get(body_mode)

        if body_range is not None:
            for frame in landmarks:
                frame_keypoints = []
                if frame:
                    json_frame = json.loads(frame)
                    for i in body_range:
                        print(json_frame[i])
                        x_coor = json_frame[i]['x']*self.config.metadata['frame_width']
                        y_coor = json_frame[i]['y']*self.config.metadata['frame_height']

                        if ((x_coor <= self.config.metadata['frame_width']) and (y_coor <= self.config.metadata['frame_height'] )):
                            frame_keypoints.append({
                                'X': x_coor,
                                'Y': y_coor,
                                })
                    
                    x_coords = [point['X'] for point in frame_keypoints]
                    y_coords = [point['Y'] for point in frame_keypoints]

                    result.append({
                        'minX': min(x_coords),
                        'minY': min(y_coords),
                        'maxX': max(x_coords),
                        'maxY': max(y_coords),
                        })
                else:
                    result.append({})
        self.bbox = result
        with open(self.config.get_log_path(custom_name=str(body_mode)), "w") as outfile:
            outfile.write(json.dumps(result))

    def process(self):
        
        if self.config.body != 'full_body':
            self.process_body_dimensions(self.config.body)
        else:
            with open(self.config.get_log_path(include_detector=True), 'r') as file:
                self.bbox = json.load(file)
                
            
        if not os.path.exists(self.config.get_log_path(custom_name=str(self.config.movement))):
            self.get_video_dimensions()
            getattr(self, self.config.movement)()

            self.analyze_movement_ranges(self.fbox)

            with open(self.config.get_log_path(custom_name=str(self.config.movement)), "w") as outfile:
                outfile.write(json.dumps(self.fbox))
        