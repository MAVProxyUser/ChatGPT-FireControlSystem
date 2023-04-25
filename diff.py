import numpy as np
import cv2
import depthai as dai
import time
from typing import List

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

p = dai.Pipeline()
p.setOpenVINOVersion(dai.OpenVINO.VERSION_2021_4)

camRgb = p.create(dai.node.ColorCamera)
camRgb.setVideoSize(720, 720)
camRgb.setPreviewSize(720, 720)
camRgb.setInterleaved(False)

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

rgb_xout = p.create(dai.node.XLinkOut)
rgb_xout.setStreamName("rgb")
camRgb.video.link(rgb_xout.input)

archive_size = 5
archive = []

with dai.Device(p) as device:
    qNn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
    qCam = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

    def get_frame(data: dai.NNData, shape):
        diff = np.array(data.getFirstLayerFp16()).reshape(shape)
        colorize = cv2.normalize(diff, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        return cv2.applyColorMap(colorize, cv2.COLORMAP_JET)

    prev_rois = []
    decay_time = 7
    decay_info = []

    # Initialize Kalman filter
    kf = create_kalman_filter()

    while True:
        diff_frame = get_frame(qNn.get(), (720, 720))
        video_frame = qCam.get().getCvFrame()
        current_time = time.time()

        rois = extract_regions_of_interest(diff_frame)

        # Update the archive
        archive.append(rois)
        if len(archive) > archive_size:
            archive.pop(0)

        # Estimate the center of the larger object
        larger_object_center = np.zeros(2)
        roi_count = 0
        for frame_rois in archive:
            for x, y, w, h in frame_rois:
                larger_object_center += np.array([x + w / 2, y + h / 2])
                roi_count += 1

        if roi_count > 0:
            larger_object_center /= roi_count
            cv2.circle(video_frame, tuple(larger_object_center.astype(int)), 5, (0, 255, 255), -1)

        # Process regions of interest and update decay_info
        decay_info = [(roi, *process_motion_data(prev_roi, roi, 720, 720), current_time) for prev_roi, roi in zip(prev_rois, rois) if prev_rois]

        # Decay and display information
        decay_info = [(roi, azimuth, velocity, direction, timestamp) for roi, azimuth, velocity, direction, timestamp in decay_info if current_time - timestamp < decay_time]

        for roi, info in zip(rois, decay_info):
            x, y, w, h = roi
            corrected_center = kf.correct(np.array([x + w / 2, y + h / 2], dtype=np.float32))
            corrected_x, corrected_y = int(corrected_center[0] - w / 2), int(corrected_center[1] - h / 2)

            # Draw the corrected rectangle
            cv2.rectangle(video_frame, (corrected_x, corrected_y), (corrected_x + w, corrected_y + h), (0, 255, 0), 2)

            # Predict next state
            predicted_center = kf.predict()
            predicted_x, predicted_y = int(predicted_center[0] - w / 2), int(predicted_center[1] - h / 2)

            # Draw the predicted rectangle
            cv2.rectangle(video_frame, (predicted_x, predicted_y), (predicted_x + w, predicted_y + h), (255, 0, 0), 2)

            # Display Azimuth, Velocity, and Direction on the video frame
            azimuth, velocity, direction, _ = info[1:]
            cv2.putText(video_frame, f"Azimuth: {azimuth:.2f}", (predicted_x, predicted_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(video_frame, f"Velocity: {velocity:.2f}", (predicted_x, predicted_y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(video_frame, f"Direction: {direction:.2f}", (predicted_x, predicted_y - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        prev_rois = rois

        cv2.imshow("Diff", diff_frame)
        cv2.imshow("Color", video_frame)

        if cv2.waitKey(1) == ord('q'):
            break


