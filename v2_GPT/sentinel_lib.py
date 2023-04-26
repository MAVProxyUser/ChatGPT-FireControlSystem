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

# Initialize AprilTag detector
detector = apriltag.Detector()

# Initialize Luxonis OAK-1-MAX pipeline
pipeline = dai.Pipeline()

# Color camera
cam = pipeline.createColorCamera()
cam.setPreviewSize(640, 480)
cam.setInterleaved(False)
cam.setFps(30)
xout_color = pipeline.createXLinkOut()
xout_color.setStreamName("color")
cam.preview.link(xout_color.input)

# Mono cameras
left = pipeline.createMonoCamera()
left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
left.setBoardSocket(dai.CameraBoardSocket.LEFT)
xout_left = pipeline.createXLinkOut()
xout_left.setStreamName("left")
left.out.link(xout_left.input)

right = pipeline.createMonoCamera()
right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
xout_right = pipeline.createXLinkOut()
xout_right.setStreamName("right")
right.out.link(xout_right.input)

# Create device and start the pipeline
device = dai.Device(pipeline)

color_queue = device.getOutputQueue(name="color", maxSize=1, blocking=False)
left_queue = device.getOutputQueue(name="left", maxSize=1, blocking=False)
right_queue = device.getOutputQueue(name="right", maxSize=1, blocking=False)

while True:
    # Get the current frame from the color camera
    color_frame = color_queue.get().getCvFrame()

    # Process the color frame
    process_frame(color_frame, 'Color Camera')

    # Get the current frame from the left and right mono cameras
    left_frame = left_queue.get().getCvFrame()
    right_frame = right_queue.get().getCvFrame()

    # Preview the mono camera frames
    cv2.imshow('Left Camera', left_frame)
    cv2.imshow('Right Camera', right_frame)

    # Exit the loop if 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
device.close()
device = None

