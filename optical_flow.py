import numpy as np
import cv2
import matplotlib as plt

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

video_path = 'not-processed/20240405_163216.mp4'
# for a in range(150, 200):
#     input = load_frame(video_path, a)
#     cv2.imwrite(f'experiments/frame_{a}.jpg', input)

save_all_frames(video_path)
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