import numpy as np
import cv2
import matplotlib as plt
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
import json

def flow_corr(L1, L2, d, w):
    # Ensure that the images are in float64 for better range during calculations
    L1 = L1.astype(np.float64)
    L2 = L2.astype(np.float64)

    rows, columns = L1.shape
    u = np.zeros((rows, columns), dtype=np.float64)
    v = np.zeros((rows, columns), dtype=np.float64)

    # Consider each pixel, within bounds defined by w and d
    for y1 in range(w + d + 1, rows - w - d):
        for x1 in range(w + d + 1, columns - w - d):
            min_val = float('inf')
            dx, dy = 0, 0

            # Explore all displacement positions
            for y2 in range(y1 - d, y1 + d + 1):
                for x2 in range(x1 - d, x1 + d + 1):
                    sum_sq = 0

                    # Calculate the sum of squared differences
                    for j in range(-w, w + 1):
                        for i in range(-w, w + 1):
                            diff = L1[y1 + j, x1 + i] - L2[y2 + j, x2 + i]
                            sum_sq += diff ** 2

                    # Keep the minimum error and corresponding displacement
                    if sum_sq < min_val:
                        min_val = sum_sq
                        dx, dy = x2 - x1, y2 - y1

            u[y1, x1] = dx
            v[y1, x1] = dy

    return u, v

def load_frame(video_path, frame_number):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if ret:
        return frame
    cap.release()

def save_all_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(f'experiments/frame_{frame_index}.jpg', frame)
        else:
            break
        frame_index += 1
    cap.release()

def get_mask(frame1, frame2, kernel=np.array((9,9), dtype=np.uint8)):
    """ Obtains image mask
        Inputs: 
            frame1 - Grayscale frame at time t
            frame2 - Grayscale frame at time t + 1
            kernel - (NxN) array for Morphological Operations
        Outputs: 
            mask - Thresholded mask for moving pixels
        """

    frame_diff = cv2.subtract(frame2, frame1)

    # blur the frame difference
    frame_diff = cv2.medianBlur(frame_diff, 3)
    
    mask = cv2.adaptiveThreshold(frame_diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY_INV, 11, 3)

    mask = cv2.medianBlur(mask, 3)

    # morphological operations
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return mask

def get_contour_detections(mask, thresh=400):
    contours, _ = cv2.findContours(mask, 
                                   cv2.RETR_EXTERNAL, # cv2.RETR_TREE, 
                                   cv2.CHAIN_APPROX_TC89_L1)
    detections = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        area = w*h
        if area > thresh: 
            detections.append([x,y,x+w,y+h, area])

    return np.array(detections)

def remove_contained_bboxes(boxes):
    """ Removes all smaller boxes that are contained within larger boxes.
        Requires bboxes to be soirted by area (score)
        Inputs:
            boxes - array bounding boxes sorted (descending) by area 
                    [[x1,y1,x2,y2]]
        Outputs:
            keep - indexes of bounding boxes that are not entirely contained 
                   in another box
        """
    check_array = np.array([True, True, False, False])
    keep = list(range(0, len(boxes)))
    for i in keep: # range(0, len(bboxes)):
        for j in range(0, len(boxes)):
            # check if box j is completely contained in box i
            if np.all((np.array(boxes[j]) >= np.array(boxes[i])) == check_array):
                try:
                    keep.remove(j)
                except ValueError:
                    continue
    return keep

def non_max_suppression(boxes, scores, threshold=1e-1):
    # Sort the boxes by score in descending order
    boxes = boxes[np.argsort(scores)[::-1]]

    # remove all contained bounding boxes and get ordered index
    order = remove_contained_bboxes(boxes)

    keep = []
    while order:
        i = order.pop(0)
        keep.append(i)
        for j in order:
            # Calculate the IoU between the two boxes
            intersection = max(0, min(boxes[i][2], boxes[j][2]) - max(boxes[i][0], boxes[j][0])) * \
                           max(0, min(boxes[i][3], boxes[j][3]) - max(boxes[i][1], boxes[j][1]))
            union = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]) + \
                    (boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1]) - intersection
            iou = intersection / union

            # Remove boxes with IoU greater than the threshold
            if iou > threshold:
                order.remove(j)
                
    return boxes[keep]

def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image

def create_segm_mask_image(path, path_out):
    base_options = python.BaseOptions(model_asset_path='assets/pose_landmarker.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=True)
    detector = vision.PoseLandmarker.create_from_options(options)

    # STEP 3: Load the input image.
    image = mp.Image.create_from_file(path)

    # STEP 4: Detect pose landmarks from the input image.
    detection_result = detector.detect(image)

    # STEP 5: Process the detection result. In this case, visualize it.
    annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
    cv2.imwrite("ing.jpg",cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

    segmentation_mask = detection_result.segmentation_masks[0].numpy_view()
    visualized_mask = np.repeat(segmentation_mask[:, :, np.newaxis], 3, axis=2) * 255

    ones_indices = np.where(segmentation_mask == 1)
    minY = np.min(ones_indices[0])
    minX = np.min(ones_indices[1])
    maxY = np.max(ones_indices[0])
    maxX = np.max(ones_indices[1])
    image = cv2.imread(path,1)
    cv2.imwrite("mmm.jpg",visualized_mask)

    gray_mask = visualized_mask[:, :, 0]  # Take one channel, as all channels are the same

    contours, _ = cv2.findContours(gray_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.rectangle(visualized_mask, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Draw green bounding box
    cv2.imwrite("bbxo1.jpg",visualized_mask)

    visualized_mask_image = cv2.rectangle(visualized_mask, (x, y), (x+w, y+h), (255, 0, 0), 3)
    cv2.imwrite("bbxo.jpg",visualized_mask_image)

    annotated_image = draw_landmarks_on_image(visualized_mask, detection_result)
    cv2.imwrite("bbxo546.jpg",cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))



    # for i in range(segmentation_mask.shape[0]):
    #     for j in range(segmentation_mask.shape[1]):
    #         value = segmentation_mask[i, j]
    #         if (value == 1.0):
    #            print("yeah")

    image = cv2.imread(path,1)
    landmarksX = []
    landmarksY = []

    landmarksX.sort()
    landmarksY.sort()
    hey = cv2.rectangle(visualized_mask_image, (x, y), (x+w, int(landmarksY[-1])), (255, 0, 0), 3)
    cv2.imwrite("bbxooo.jpg", hey)

def all_fboxes_one_picture(path, path_json, additional_path=None):
    with open(path_json, 'r') as file:
        data = json.load(file)
    img = load_frame(video_path, 1000)

    for a in range (0, len(data), 30):
        if data[a]:
            hey = cv2.rectangle(img, (int(data[a]['minX']), int(data[a]['minY'])), (int(data[a]['maxX']), int(data[a]['maxY'])), (255, 0, 0), 3)
    
    if additional_path:
        with open(additional_path, 'r') as file:
            data = json.load(file)
        hey = cv2.rectangle(img, (int(data[0]['minX']), int(data[0]['minY'])), (int(data[0]['maxX']), int(data[0]['maxY'])), (0, 0, 255), 3)
    cv2.imwrite('annotated_fboxs.jpg', hey)


video_path = 'not-processed/curling1.mp4'
json_path = "log/acroyoga3/yolo.json"
additional_path = "log/acroyoga3/pose_anchor_fullbox_minimal.json"
# for a in range(0, 100):
#     input = load_frame(video_path, a)
#     cv2.imwrite(f'experiments/frame_{a}.jpg', input)

input = load_frame(video_path, 200)
cv2.imwrite(f'experiments/frame_200.jpg', input)


# all_fboxes_one_picture(video_path, json_path, additional_path)
# image_path = "experiments/00_dancing3_frame_507.jpg"
# path_out = "experiments/00_dancing3_frame_507_out.jpg"
# create_segm_mask_image(image_path, path_out)
# save_all_frames(video_path)
# input = load_frame(video_path, 5)
# inputimage2 = load_frame(video_path, 6)

# img1 = cv2.cvtColor(inputimage1, cv2.COLOR_RGB2RGBA)
# img2 = cv2.cvtColor(inputimage2, cv2.COLOR_RGB2RGBA)
# cv2.imwrite('experiments/frame1.jpg', img1)
# cv2.imwrite('experiments/frame2.jpg', img2)


# grayscale_diff = cv2.subtract(img2, img1)
# cv2.imwrite('experiments/grayscale_diff.jpg', grayscale_diff)

# kernel = np.array((9,9), dtype=np.uint8)
# mask = get_mask(img1, img2, kernel)
# cv2.imwrite('experiments/mask.jpg', mask)

# # separate bboxes and scores
# bboxes = detections[:, :4]
# scores = detections[:, -1]

# # Get Non-Max Suppressed Bounding Boxes
# nms_bboxes = non_max_suppression(bboxes, scores, threshold=0.1)