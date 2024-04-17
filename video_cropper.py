import json

class VideoCropper:
    def __init__(self, crop_mode):
        self.crop_mode = crop_mode
    
    def crop(self, video_path, json_path):
        coordinates = self.load_from_json(json_path)
        # Your video cropping code here, using the loaded coordinates
    
    def load_from_json(self, file_path):
        with open(file_path) as json_file:
            data = json.load(json_file)
        return data