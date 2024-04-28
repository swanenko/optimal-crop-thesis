import json
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from video_detector import VideoDetector
from collections import Counter
from scipy.signal import savgol_filter
from fractions import Fraction

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

    def detect_and_correct_consecutive_outliers(self, data, threshold, set):
        print(set)
        print(threshold)
        corrected_data = data.copy() 
        for i in range(1, len(data)):
            if abs(data[i] - data[i - 1]) > threshold/2:
                if (abs(data[i] - data[i + 1]) > threshold/2):
                    print(data[i - 1], " ", data[i], data[i+1])
                    corrected_data[i] = corrected_data[i - 1] 
        return corrected_data
    
    def analyze_differences(self, data, set):
        differences_x1_x = [abs(data[i] - data[i-1]) for i in range(1, len(data))]
        differences_x_x1 = [abs(data[i+1] - data[i]) for i in range(len(data) - 1)]
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.boxplot(data=differences_x1_x)
        plt.title('Differences between x-1 and x')
        plt.subplot(1, 2, 2)
        sns.boxplot(data=differences_x_x1)
        plt.title('Differences between x and x+1')
        plt.suptitle('Analysis of Differences to Determine Threshold')
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.hist(differences_x1_x, bins=30, color='skyblue', alpha=0.7)
        plt.title('Histogram of differences between x-1 and x')
        plt.subplot(1, 2, 2)
        plt.hist(differences_x_x1, bins=30, color='lightgreen', alpha=0.7)
        plt.title('Histogram of differences between x and x+1')
        plt.suptitle('Histogram Analysis of Differences')
        plt.savefig(f'boxplot_differences_{set}.png')

    def filter_outliers(self, list):
        minX = [bbox['minX'] for bbox in list]
        minY = [bbox['minY'] for bbox in list]
        width = [bbox['maxX'] - bbox['minX'] for bbox in list]
        height = [bbox['maxY'] - bbox['minY'] for bbox in list]

        corrected_minX = self.detect_and_correct_consecutive_outliers(minX, self.dimensions[0] * 0.1, "X")
        corrected_minY = self.detect_and_correct_consecutive_outliers(minY, self.dimensions[1] * 0.1, "Y")

        corrected_maxX = [corrected_minX[i] + width[i] for i in range(len(width))]
        corrected_maxY = [corrected_minY[i] + height[i] for i in range(len(height))]

        updated_bboxes = [{'minX': corrected_minX[i], 'minY': corrected_minY[i],
                        'maxX': corrected_maxX[i], 'maxY': corrected_maxY[i]}
                        for i in range(len(corrected_minX))]

        return updated_bboxes

    def find_extremes(self, list):
        if not list:
            return None  

        min_x = float('inf')
        max_x = float('-inf')
        min_y = float('inf')
        max_y = float('-inf')

        for bbox in list:
            min_x = min(min_x, bbox['minX'], bbox['maxX'])
            max_x = max(max_x, bbox['minX'], bbox['maxX'])

            min_y = min(min_y, bbox['minY'], bbox['maxY'])
            max_y = max(max_y, bbox['minY'], bbox['maxY'])

        return (min_x, max_x, min_y, max_y)

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

    def savgol_stabilize(self,  data):
        iterations = len(data)
        result = []

        min_x_values = [fbox['minX'] for fbox in data]
        min_y_values = [fbox['minY'] for fbox in data]
        
        yhat_min_x = savgol_filter(min_x_values, 51, 3)
        yhat_min_y = savgol_filter(min_y_values, 51, 3) 

        for i in range(iterations):

            result.append({
                'minX': int(yhat_min_x[i]),
                'minY': int(yhat_min_y[i]),
                'maxX': int(yhat_min_x[i]+self.dimensions[0]),
                'maxY': int(yhat_min_y[i]+self.dimensions[1]),
            })
        return result
    
    def adjust_frame_size(self, list):
        for box in list:
            if box['minX'] < 0:
                box['minX'] = 0
                box['maxX'] = self.dimensions[0]  
            elif box['maxX'] > self.config.metadata['frame_width']:
                box['maxX'] = self.config.metadata['frame_width']
                box['minX'] = self.config.metadata['frame_width'] - self.dimensions[0]  

            if box['minY'] < 0:
                box['minY'] = 0
                box['maxY'] = self.dimensions[1]  
            elif box['maxY'] > self.config.metadata['frame_height']:
                box['maxY'] = self.config.metadata['frame_height']
                box['minY'] = self.config.metadata['frame_height'] - self.dimensions[1]  
            
        return list

    def zoom(self):
        if self.config.size == 'minimal':
            fraction = Fraction(self.dimensions[0], self.dimensions[1])
            aspect_ratio = (fraction.numerator, fraction.denominator)
        else:
            aspect_ratio = self.config.size

        factor=2
        for bbox in self.bbox:
            bbox_width = bbox['maxX'] - bbox['minX']
            bbox_height = bbox['maxY'] - bbox['minY']
          
            center_x = bbox['minX'] + (bbox_width//2)
            center_y = bbox['minY'] + (bbox_height//2)

            if bbox_width / bbox_height > aspect_ratio[0] / aspect_ratio[1]:
                cropped_height = max(bbox_height * factor, bbox_width * factor * aspect_ratio[1] / aspect_ratio[0])
                cropped_width = cropped_height * aspect_ratio[0] / aspect_ratio[1]
            else:
                cropped_width = max(bbox_width * factor, bbox_height * factor * aspect_ratio[0] / aspect_ratio[1])
                cropped_height = cropped_width * aspect_ratio[1] / aspect_ratio[0]

            frame_min_x = center_x - cropped_width / 2
            frame_min_y = center_y - cropped_height / 2
            frame_max_x = center_x + cropped_width / 2
            frame_max_y = center_y + cropped_height / 2

            self.fbox.append({
                'minX': frame_min_x,
                'minY': frame_min_y,
                'maxX': frame_max_x,
                'maxY': frame_max_y,
            })

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

    def horizontal(self):
        min_y_values = [bbox['minY'] for bbox in self.bbox]
        max_y_values = [bbox['maxY'] for bbox in self.bbox]
        
        global_min_y = min(min_y_values)
        global_max_y = max(max_y_values)

        self.dimensions = (self.dimensions[0], global_max_y - global_min_y)

        for bbox in self.bbox:
            bbox_width = bbox['maxX'] - bbox['minX']
            center_x = bbox['minX'] + (bbox_width // 2)

            frame_min_x = center_x - (self.dimensions[0] // 2)
            frame_max_x = frame_min_x + self.dimensions[0]

            self.fbox.append({
                'minX': frame_min_x,
                'minY': global_min_y,  
                'maxX': frame_max_x,
                'maxY': global_max_y 
            })

    def vertical(self):
        min_x_values = [bbox['minX'] for bbox in self.bbox]
        max_x_values = [bbox['maxX'] for bbox in self.bbox]

        global_min_x = min(min_x_values)
        global_max_x = max(max_x_values)

        self.dimensions = (global_max_x - global_min_x, self.dimensions[1])

        for bbox in self.bbox:
            bbox_height = bbox['maxY'] - bbox['minY']
            center_y = bbox['minY'] + (bbox_height // 2)

            frame_min_y = center_y - (self.dimensions[1] // 2)
            frame_max_y = frame_min_y + self.dimensions[1]

            self.fbox.append({
                'minX': global_min_x,  
                'minY': frame_min_y,
                'maxX': global_max_x, 
                'maxY': frame_max_y
            })

    def fullbox(self):
        extremes = self.find_extremes(self.bbox)
        self.fbox.append({
        'minX': extremes[0], 
        'minY': extremes[2],
        'maxX': extremes[1],  
        'maxY': extremes[3]
        })

    def get_video_dimensions(self):
        widths = []
        heights = []
        for point in self.bbox:
            if point:
                widths.append(point['maxX']-point['minX'])
                heights.append(point['maxY']-point['minY'])

        self.dimensions = (max(widths), max(heights))

    def transform_dimensions_closest(x, y, aspect_ratio):
        scale_width = (y * aspect_ratio[0]) / aspect_ratio[1]
        scale_height = (x * aspect_ratio[1]) / aspect_ratio[0]

        if abs(scale_width - x) < abs(scale_height - y):
            return (int(scale_width), y)
        else:
            return (x, int(scale_height))

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

    def normalize_bbox(self):
        last_non_empty_bbox = None  
        for i in range(len(self.bbox)):
            if not self.bbox[i]:  
                for j in range(i + 1, len(self.bbox)):
                    if self.bbox[j]:
                        self.bbox[i] = self.bbox[j].copy()  
                        break
                else:
                    if last_non_empty_bbox is not None:
                        self.bbox[i] = last_non_empty_bbox.copy()
                    else:
                        print("Bbox normalization error")
                        pass
            else:
                last_non_empty_bbox = self.bbox[i] 

    def process(self):
        
        if self.config.body != 'full_body':
            self.process_body_dimensions(self.config.body)
        else:
            with open(self.config.get_log_path(include_detector=True), 'r') as file:
                self.bbox = json.load(file)
        
        self.normalize_bbox()
        
        self.get_video_dimensions()  
            
        if not os.path.exists(self.config.get_log_path(include_fbox=True)):

            if 'zoom' in self.config.process:
                self.zoom()
            else:
                path = f"{self.config.movement}"
                
                if self.config.size != 'minimal':
                    self.dimensions = self.transform_dimensions_closest(*self.dimensions, aspect_ratio=self.config.size)
            
                getattr(self, path)()
                self.analyze_movement_ranges(self.fbox)

                if 'no_outliers' in self.config.process:
                    self.fbox = self.filter_outliers(self.fbox)

            if 'stable' in self.config.process:
                self.fbox = self.savgol_stabilize(self.fbox)
                
            self.fbox = self.adjust_frame_size(self.fbox)

            with open(self.config.get_log_path(include_fbox=True), "w") as outfile:
                outfile.write(json.dumps(self.fbox))
            
            with open(self.config.get_log_path(include_stats=True), "r") as stat_file:
                stats = json.load(stat_file)
            
            if 'fullbox' in self.config.movement:
                stats['out_width'] = self.fbox[0]['maxX'] - self.fbox[0]['minX']
                stats['out_height'] = self.fbox[0]['maxY'] - self.fbox[0]['minY']
            else:
                stats['out_width'] = self.dimensions[0]
                stats['out_height'] = self.dimensions[1]

            with open(self.config.get_log_path(include_stats=True), 'w') as file:
                json.dump(stats, file, indent=4)
            