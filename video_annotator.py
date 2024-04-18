import json
import cv2
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

def deserialize_landmarks(frame_data):
    landmark_list = landmark_pb2.NormalizedLandmarkList()
    frame_data = json.loads(frame_data)
    for landmark_dict in frame_data:
        landmark = landmark_list.landmark.add()
        landmark.x = landmark_dict['x']
        landmark.y = landmark_dict['y']
        landmark.z = landmark_dict['z']
        if 'visibility' in landmark_dict:
            landmark.visibility = landmark_dict['visibility']
    return landmark_list

def draw_landmarks_on_image(rgb_image, detection_result):
    if not detection_result:
        return rgb_image
    
    annotated_image = np.copy(rgb_image)
    pose_landmarks = json.loads(detection_result)

    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
    landmark_pb2.NormalizedLandmark(x=landmark['x'], y=landmark['y'], z=landmark['z']) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
    annotated_image,
    pose_landmarks_proto,
    solutions.pose.POSE_CONNECTIONS,
    solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image

class VideoAnnotator:
    def __init__(self, config):
        self.config = config
        self.draw_strategy = {}
        self.annotation_data = {}
        self.fullbox = None


    def draw_bbox(self, frame, bbox, color=(255, 0, 0), thickness=3):
        if not bbox:
            pass
        else:
            cv2.rectangle(frame, (int(bbox['minX']), int(bbox['minY'])), 
                        (int(bbox['maxX']), int(bbox['maxY'])), color, thickness)
        return frame
    
    def draw_fullbox(self, frame, color=(0, 0, 255), thickness=3):
        cv2.rectangle(frame, (int(self.fullbox[0]['minX']), int(self.fullbox[0]['minY'])), 
                    (int(self.fullbox[0]['maxX']), int(self.fullbox[0]['maxY'])), color, thickness)
        return frame

    def draw_landmarks(self, frame, landmarks):
        frame = draw_landmarks_on_image(frame, landmarks)
        return frame

    def draw_fbox(self, frame, bbox, color=(0, 255, 0), thickness=3):
        cv2.rectangle(frame, (int(bbox['minX']), int(bbox['minY'])), 
                      (int(bbox['maxX']), int(bbox['maxY'])), color, thickness)
        return frame

    def setup_annotation_strategy(self):
        for annotation_type in self.config.annotate:
            if annotation_type == 'landmarks':
                with open(self.config.get_log_path(custom_name="landmarks"), 'r') as file:
                    self.annotation_data[annotation_type] = json.load(file)
                self.draw_strategy[annotation_type] = self.draw_landmarks

            elif annotation_type == 'bbox':
                with open(self.config.get_log_path(include_detector=True), 'r') as file:
                    self.annotation_data[annotation_type] = json.load(file)
                self.draw_strategy[annotation_type] = self.draw_bbox

            elif annotation_type == 'fbox':
                with open(self.config.get_log_path(include_fbox=True), 'r') as file:
                    self.annotation_data[annotation_type] = json.load(file)
                if (len(self.annotation_data[annotation_type]) == 1):
                    self.fullbox = self.annotation_data[annotation_type]
                self.draw_strategy[annotation_type] = self.draw_fbox
            
            else:
                print('No such annotation type:', annotation_type)
                pass

        if  self.config.body != 'full_body':
            with open(self.config.get_log_path(custom_name=str(self.config.body)), 'r') as file:
                self.annotation_data[annotation_type] = json.load(file)
            self.draw_strategy[annotation_type] = self.draw_bbox

    def process(self):

        self.setup_annotation_strategy()

        cap = cv2.VideoCapture(self.config.src)
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter(self.config.get_video_path(annotate=True), fourcc, self.config.metadata['fps'], (int(self.config.metadata['frame_width']), int(self.config.metadata['frame_height'])))
        frame_index = 0

        if not cap.isOpened():
            print(f"Error opening video file: {self.config.src}")
            return
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                for annotation_type in self.config.annotate:
                    if frame_index < len(self.annotation_data[annotation_type]):
                        frame = self.draw_strategy[annotation_type](frame, self.annotation_data[annotation_type][frame_index])
                if self.fullbox:
                    frame = self.draw_fullbox(frame)

                out.write(frame)
            else:
                break
            frame_index += 1
        cap.release()
        out.release()

