import numpy as np
import cv2
import depthai as dai
import time
from typing import List, Tuple
from functools import partial

def get_roi_area(roi):
    _, _, w, h = roi
    return w * h

def limit_lead_distance(yellow_x, yellow_y, green_x, green_y, max_distance=50):
    dx = green_x - yellow_x
    dy = green_y - yellow_y
    distance = np.sqrt(dx ** 2 + dy ** 2)

    if distance > max_distance:
        green_x = yellow_x + (dx / distance) * max_distance
        green_y = yellow_y + (dy / distance) * max_distance

    return green_x, green_y

def get_interception_point(x: float, y: float, velocity: float, direction: float, lead_time: float) -> Tuple[float, float]:
    direction_rad = np.deg2rad(direction)
    delta_x = velocity * np.cos(direction_rad) * lead_time
    delta_y = velocity * np.sin(direction_rad) * lead_time
    interception_x = x + delta_x
    interception_y = y + delta_y
    return interception_x, interception_y

def extract_regions_of_interest(diff, threshold=0.2, min_side_length=25):
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(gray_diff, threshold * 255, 255, cv2.THRESH_BINARY)
    mask = mask.astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_roi = None
    largest_area = 0

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        if w >= min_side_length and h >= min_side_length:
            area = w * h
            if area > largest_area:
                largest_area = area
                largest_roi = (x, y, w, h)

    return [largest_roi] if largest_roi else []

def create_kalman_filter() -> cv2.KalmanFilter:
    kf = cv2.KalmanFilter(4, 2)
    kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)
    kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)
    kf.processNoiseCov = np.array([[1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32) * 1e-2
    kf.measurementNoiseCov = np.array([[1, 0],
                                       [0, 1]], np.float32) * 1e-4
    kf.errorCovPost = np.eye(4, dtype=np.float32)
    kf.statePost = np.zeros(4, dtype=np.float32)
    return kf

def get_center(x: float, y: float, w: float, h: float) -> List[float]:
    return [x + w / 2, y + h / 2, 1]

def process_motion_data(prev_roi, curr_roi, frame_width, frame_height):
    prev_x, prev_y, prev_w, prev_h = prev_roi
    curr_x, curr_y, curr_w, curr_h = curr_roi

    x_center = frame_width / 2
    y_center = frame_height / 2

    # Calculate azimuth
    x_diff = (curr_x + curr_w / 2) - x_center
    y_diff = (curr_y + curr_h / 2) - y_center
    azimuth = np.arctan2(y_diff, x_diff) * 180 / np.pi

    # Calculate velocity and direction
    delta_x = curr_x - prev_x
    delta_y = curr_y - prev_y
    velocity = np.sqrt(delta_x ** 2 + delta_y ** 2)
    direction = np.arctan2(delta_y, delta_x) * 180 / np.pi

    return azimuth, velocity, direction

def draw_depth_circle(depth_frame: np.ndarray, video_frame: np.ndarray, position: Tuple[int, int], radius: int, color: Tuple[int, int, int], thickness: int) -> None:
    x, y = position
    depth = depth_frame[y, x]
    if depth != 0:
        cv2.circle(video_frame, (x, y), radius, color, thickness)

def get_frame(data: dai.NNData, shape):
    diff = np.array(data.getFirstLayerFp16()).reshape(shape)
    colorize = cv2.normalize(diff, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
    return cv2.applyColorMap(colorize, cv2.COLORMAP_JET)

p = dai.Pipeline()

# Create a source - color camera
camRgb = p.create(dai.node.ColorCamera)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setVideoSize(720, 720)
camRgb.setPreviewSize(720, 720)
camRgb.setInterleaved(False)

# Add NN creation back to the pipeline
nn = p.create(dai.node.NeuralNetwork)
nn.setBlobPath("models/diff_openvino_2021.4_6shave.blob")

script = p.create(dai.node.Script)
camRgb.preview.link(script.inputs['in'])
script.setScript("""
old = node.io['in'].get()
while True:
    frame = node.io['in'].get()
    node.io['img1'].send(old)
    node.io['img2'].send(frame)
    old = frame
""")
script.outputs['img1'].link(nn.inputs['img1'])
script.outputs['img2'].link(nn.inputs['img2'])

nn_xout = p.create(dai.node.XLinkOut)
nn_xout.setStreamName("nn")
nn.out.link(nn_xout.input)

# Create an xlink output for color frames
rgb_xout = p.create(dai.node.XLinkOut)
rgb_xout.setStreamName("rgb")
camRgb.video.link(rgb_xout.input)

# Add depth to the pipeline
monoLeft = p.create(dai.node.MonoCamera)
monoRight = p.create(dai.node.MonoCamera)
depth = p.create(dai.node.StereoDepth)
xlink_out_depth = p.create(dai.node.XLinkOut)

monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

depth.initialConfig.setConfidenceThreshold(200)
depth.initialConfig.setMedianFilter(dai.StereoDepthProperties.MedianFilter.KERNEL_7x7)

monoLeft.out.link(depth.left)
monoRight.out.link(depth.right)

depth.disparity.link(xlink_out_depth.input)
xlink_out_depth.setStreamName("depth")

with dai.Device(p) as device:
    qNn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
    qCam = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    qDepth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

    archive = []
    archive_size = 5

    prev_rois = []
    decay_time = 3
    decay_info = []
    interception_points = []

    # Initialize Kalman filter
    kf = create_kalman_filter()

    frame_number = 0
    WINDOW_WIDTH = 720
    WINDOW_HEIGHT = 720

    while True:
        depth_frame = qDepth.get().getFrame()
        diff_frame = get_frame(qNn.get(), (720, 720))
        video_frame = qCam.get().getCvFrame()
        current_time = time.time()

        rois = extract_regions_of_interest(diff_frame)

        # Update the archive
        archive.append(rois)
        if len(archive) > archive_size:
            archive.pop(0)

        n_largest_rois = 3  # Change this value to consider more or fewer top ROIs

        # Estimate the center of the larger object
        larger_object_center = np.zeros(2)
        roi_count = 0
        for frame_rois in archive:
            sorted_rois = sorted(frame_rois, key=get_roi_area, reverse=True)[:n_largest_rois]
            for x, y, w, h in sorted_rois:
                larger_object_center += np.array([x + w / 2, y + h / 2])
                roi_count += 1

        if roi_count > 0:
            larger_object_center /= roi_count
            cv2.circle(video_frame, tuple(larger_object_center.astype(int)), 5, (0, 255, 255), -1)

        # Process regions of interest and update decay_info
        decay_info = [(roi, *process_motion_data(prev_roi, roi, 720, 720), current_time) for prev_roi, roi in zip(prev_rois, rois) if prev_roi is not None]

        for roi, azimuth, velocity, direction, roi_time in decay_info:
            x, y, w, h = roi
            center = get_center(x, y, w, h)
            lead_time = 1  # Change this value to adjust the distance between the yellow and green dots

            # Get the interception point and limit its distance from the yellow dot
            interception_x, interception_y = get_interception_point(larger_object_center[0], larger_object_center[1], velocity, direction, lead_time)
            interception_x, interception_y = limit_lead_distance(larger_object_center[0], larger_object_center[1], interception_x, interception_y)

            # Update the interception point in the interception_points list
            interception_points.append((frame_number, interception_x, interception_y))

            # Draw the green dot for the interception point
            draw_depth_circle(depth_frame, video_frame, (int(interception_x), int(interception_y)), 5, (0, 255, 0), -1)

        prev_rois = rois

        # Flip images horizontally
        diff_frame = cv2.flip(diff_frame, 1)
        video_frame = cv2.flip(video_frame, 1)

        cv2.imshow("Diff", diff_frame)
        cv2.imshow("Color", video_frame)

        # Move windows
        cv2.moveWindow("Diff", 0, 0)
        cv2.moveWindow("Color", WINDOW_WIDTH , 0)

        frame_number += 1

        key = cv2.waitKey(1)
        if key == 27 or key == ord("q"):
            break

cv2.destroyAllWindows()

