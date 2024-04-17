import json
from video_configuration import Config
import os
import cv2
import mediapipe as mp
import json
import os
import os.path as op
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import numpy as np

class VideoDetector:
    def __init__(self, config):
        self.config = config
        self.path = self.config.get_log_path(include_detector=True)
        self.video_keypoints = []
        self.start_frame = 0
        self.frame = 0
    
    def existing_detecion(self):
        return os.path.exists(self.path)

    def load_from_json(self, file_path):
        with open(file_path) as json_file:
            data = json.load(json_file)
    
    def process_mpipelmark(self, cap):
        landmarks = []
        with mp_pose.Pose(
            min_detection_confidence=0.1,
            min_tracking_confidence=0.1) as pose:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    break

                frame_keypoints = []
                
                results = pose.process(image)
                if(results.pose_landmarks):
                    for data_point in results.pose_landmarks.landmark:

                        x_coor =  data_point.x*self.config.metadata['frame_width']
                        y_coor =  data_point.y*self.config.metadata['frame_height']

                        if ((x_coor <= self.config.metadata['frame_width']) and (y_coor <= self.config.metadata['frame_height'] )):
                            frame_keypoints.append({
                                                'X': data_point.x*self.config.metadata['frame_width'],
                                                'Y': data_point.y*self.config.metadata['frame_height'],
                                                })
                    
                    x_coords = [point['X'] for point in frame_keypoints]
                    y_coords = [point['Y'] for point in frame_keypoints]

                    self.video_keypoints.append({
                        'minX': min(x_coords),
                        'minY': min(y_coords),
                        'maxX': max(x_coords),
                        'maxY': max(y_coords),
                        })
                    landmarks.append(serialize_landmarks(results.pose_landmarks))
                else:
                    if self.frame >= 1 and (self.start_frame != self.frame-1):
                        self.video_keypoints.append(self.video_keypoints[self.frame-1])
                    elif self.start_frame == 0:
                        self.start_frame = self.frame+1
                        self.video_keypoints.append({})
                    else:
                        self.video_keypoints.append({})
                    landmarks.append([])

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
                self.frame = self.frame + 1

        with open(self.config.get_log_path(custom_name="landmarks"), 'w') as file:
            json.dump(landmarks, file, indent=4)    
          
    def process_yolo(self, cap):
        (height, width) = (self.config.metadata['frame_height'], self.config.metadata['frame_width'])
        net = cv2.dnn.readNet("assets/yolov3.weights", "assets/yolov3.cfg") # download from https://pjreddie.com/darknet/yolo/
        cap = cv2.VideoCapture(self.config.src)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
                net.setInput(blob)
                output_layer_name = net.getUnconnectedOutLayersNames()
                output_layers = net.forward(output_layer_name)
                people = []
                for output in output_layers:
                    for detection in output:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]

                        if class_id == 0 and confidence > 0.5:
                            center_x = int(detection[0] * width)
                            center_y = int(detection[1] * height)
                            w = int(detection[2] * width)
                            h = int(detection[3] * height)
                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)

                            people.append((x, y, w, h))
                            break
                if people:
                    if self.config.multiple:
                        min_x = float('inf')
                        min_y = float('inf')
                        max_x = float('-inf')
                        max_y = float('-inf')

                        for person in people:
                            min_x = min(min_x, person[0])  
                            min_y = min(min_y, person[1]) 
                            max_x = max(max_x, person[0] + person[2])  
                            max_y = max(max_y, person[1] + person[3]) 
                        
                        self.video_keypoints.append({
                                        'minX': min_x,
                                        'minY': min_y,
                                        'maxX': max_x,
                                        'maxY': max_y,
                                        })
                    else:
                        x = people[0][0]
                        y = people[0][1]
                        w = people[0][2]
                        h = people[0][3]
                        
                        self.video_keypoints.append({
                                                    'minX': x,
                                                    'minY': y,
                                                    'maxX': x+w,
                                                    'maxY': y+h,
                                                    })
                else:
                    self.video_keypoints.append({})
            else:
                break
            self.frame = self.frame + 1
            print(self.frame)
    
    def process_mpipeseg(self, cap):
        base_options = python.BaseOptions(model_asset_path='assets/pose_landmarker.task') # download from https://developers.google.com/mediapipe/solutions/vision/pose_landmarker/index#models
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=True)
        detector = vision.PoseLandmarker.create_from_options(options)

        with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
                while cap.isOpened():
                    success, frame_image = cap.read()
                    if not success:
                        break
                
                    try:
                        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_image)
                        detection_result = detector.detect(image)
                        segmentation_mask = detection_result.segmentation_masks[0].numpy_view()
                        visualized_mask = np.repeat(segmentation_mask[:, :, np.newaxis], 3, axis=2) * 255
                        ones_indices = np.where(segmentation_mask == 1)   
                        gray_mask = visualized_mask[:, :, 0]  
                        contours, _ = cv2.findContours(gray_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        largest_contour = max(contours, key=cv2.contourArea)
                        x, y, w, h = cv2.boundingRect(largest_contour)
                        self.video_keypoints.append({
                                                'minX': x,
                                                'minY': y,
                                                'maxX': x+w,
                                                'maxY': y+h,
                                                })
                    except:
                        print("Failture in detection")
                        if self.frame >= 1:
                            self.video_keypoints.append(self.video_keypoints[self.frame-1])
                        if self.start_frame == 0:
                            self.start_frame = self.frame
                        else:
                            self.video_keypoints.append({})
                        
                    self.frame = self.frame + 1

    def process(self):
        if self.existing_detecion():
            return
         
        cap = cv2.VideoCapture(self.config.src)
        if not cap.isOpened():
            print(f"Error opening video file: {self.config.src}")
            return
        
        if self.config.multiple and self.config.detector != 'yolo':
            print(f"Multiple people processing is possible only with yolo detector. Switching to yolo.")

        if self.config.detector == 'mpipeseg':
            self.process_mpipeseg(cap)
        elif self.config.detector == 'yolo':
            self.process_yolo(cap)
        elif self.config.detector == 'mpipelmark':
            self.process_mpipelmark(cap)
        else:
            print(f"Unknown detector type: {self.config.detector}")

        cap.release()

        with open(self.path, "w") as outfile:
            outfile.write(json.dumps(self.video_keypoints))

        with open(self.config.get_log_path(include_stats=True), "r") as stat_file:
            stats = json.load(stat_file)
        stats['start_frame'] = self.frame

        with open(self.config.get_log_path(include_stats=True), 'w') as file:
            json.dump(stats, file, indent=4)

def landmark_list_to_dict(landmark_list):
    return [{
        'x': landmark.x,
        'y': landmark.y,
        'z': landmark.z,
        'visibility': getattr(landmark, 'visibility', 0)  # Some landmarks may not have visibility
    } for landmark in landmark_list.landmark]

def serialize_landmarks(landmark_list):
    landmark_dict = landmark_list_to_dict(landmark_list)
    return json.dumps(landmark_dict)