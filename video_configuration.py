import os
from pathlib import Path
import cv2
import json

DEFAULT_SETTINGS = {
    'detector': 'yolo',
    'multiple': False,
    'body': 'full_body',
    'process': ['stable', 'no_outliers'],
    'movement': 'free',
    'size': 'minimal',
    'annotate' : [],
    'strategy': 'basic'
}

class Config:
    _instance = None
    src = ""
    _base_log_dir = "log" 
    _base_out_dir = "processed" 
    detector = 'yolo'
    multiple = False
    body = 'full_body'
    process = ['stable', 'no_outliers']
    movement = 'free'
    size = 'minimal'
    metadata = {}
    annotate = []

    @staticmethod
    def get_instance():
        if Config._instance is None:
            Config._instance = Config()
        return Config._instance
    
    @staticmethod
    def get_video_metadata(cls):
        log_path = cls.get_log_path(include_full_config=True)
        stats_file = Path(log_path).with_name("stats.json")

        stats_file.parent.mkdir(parents=True, exist_ok=True)

        if not stats_file.exists():
            cap = cv2.VideoCapture(cls.src)
            if not cap.isOpened():
                print(f"Error opening video file: {cls.src}")
                return

            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            cap.release()  

            cls.metadata = {
                "frame_width": frame_width,
                "frame_height": frame_height,
                "frames": frames,
                "fps": fps
            }

            with open(stats_file, "w") as f:
                json.dump(cls.metadata, f, indent=4)

        else:
            with open(stats_file, "r") as f:
                cls.metadata = json.load(f)

    @classmethod
    def initialize(cls, args, file_path):
        cls.src = file_path
        for setting in DEFAULT_SETTINGS:
            value = getattr(args, setting, DEFAULT_SETTINGS[setting])
            setattr(cls, setting, value)
        cls.get_video_metadata(cls)
        
    
    @classmethod
    def get_log_path(cls, include_detector=False, include_full_config=False, include_fbox=False,include_stats=False, custom_name=""):
        video_name = os.path.splitext(os.path.basename(cls.src))[0]
        process_settings = '_'.join(cls.process)

        if include_full_config:
            multiple_setting = 'multiple' if cls.multiple else 'single'
            configurations = f"{cls.detector}_{multiple_setting}_{cls.body}_{process_settings}_{cls.movement}_{cls.size}"
        elif include_detector:
            configurations = f"{cls.detector}"
        elif include_stats:
            configurations = "stats"
        elif include_fbox:
            configurations = f"{process_settings}_{cls.movement}_{cls.size}"
        else:
            configurations = custom_name
        
        log_file_name = f"{configurations}.json"
        return os.path.join(cls._base_log_dir, video_name, log_file_name)
    
    @classmethod
    def get_video_path(cls, include_detector=False, include_full_config=False, annotate=False, include_stats=False, custom_name=""):
        video_name = os.path.splitext(os.path.basename(cls.src))[0]
        process_settings = '_'.join(cls.process)
        multiple_setting = 'multiple' if cls.multiple else 'single'

        if include_full_config:
            configurations = f"{cls.detector}_{multiple_setting}_{cls.body}_{process_settings}_{cls.movement}_{cls.size}"
        elif annotate:
            configurations = f"annotate_{cls.detector}_{multiple_setting}_{cls.body}_{process_settings}_{cls.movement}_{cls.size}"
        elif include_detector:
            configurations = f"{cls.detector}"
        elif include_stats:
            configurations = "stats"
        else:
            configurations = custom_name
        
        log_file_name = f"{configurations}.mp4"
        video_path = os.path.join(cls._base_out_dir, video_name, log_file_name)
        os.makedirs(os.path.dirname(video_path), exist_ok=True)

        return video_path