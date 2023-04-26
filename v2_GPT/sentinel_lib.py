import depthai as dai
import cv2
import apriltag

def process_frame(frame, window_name):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = detector.detect(gray_frame)

    for detection in detections:
        cv2.circle(frame, tuple(detection.center.astype(int)), 4, (0, 255, 0), 2)
        cv2.putText(frame, str(detection.tag_id), tuple(detection.center.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow(window_name, frame)

def create_pipeline():
    # Initialize AprilTag detector
    global detector
    detector = apriltag.Detector()

    # Initialize Luxonis OAK-1-MAX pipeline
    pipeline = dai.Pipeline()

    # Create the color camera
    cam = pipeline.createColorCamera()
    cam.setPreviewSize(640, 480)
    cam.setInterleaved(False)
    cam.setFps(30)

    # Create the mono cameras
    mono_left = pipeline.createMonoCamera()
    mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
    mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)

    mono_right = pipeline.createMonoCamera()
    mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
    mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

    # Create output streams
    color_out = pipeline.createXLinkOut()
    color_out.setStreamName("color")
    cam.preview.link(color_out.input)

    left_out = pipeline.createXLinkOut()
    left_out.setStreamName("left")
    mono_left.out.link(left_out.input)

    right_out = pipeline.createXLinkOut()
    right_out.setStreamName("right")
    mono_right.out.link(right_out.input)

    return pipeline

def start_pipeline(pipeline):
    global device, color_queue, left_queue, right_queue
    device = dai.Device(pipeline)

    color_queue = device.getOutputQueue(name="color", maxSize=1, blocking=False)
    left_queue = device.getOutputQueue(name="left", maxSize=1, blocking=False)
    right_queue = device.getOutputQueue(name="right", maxSize=1, blocking=False)

def close_pipeline():
    cv2.destroyAllWindows()
    device.close()
    device = None

