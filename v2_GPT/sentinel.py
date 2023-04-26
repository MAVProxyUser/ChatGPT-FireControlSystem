import cv2
import sentinel_lib

# Create and start the pipeline
pipeline = sentinel_lib.create_pipeline()
sentinel_lib.start_pipeline(pipeline)

while True:
    # Get the current frame from the color camera
    color_frame = sentinel_lib.color_queue.get().getCvFrame()

    # Process the color frame
    sentinel_lib.process_frame(color_frame, 'Color Camera')
    cv2.moveWindow('Color Camera', 640, 0)

    # Get the current frame from the left and right mono cameras
    left_frame = sentinel_lib.left_queue.get().getCvFrame()
    right_frame = sentinel_lib.right_queue.get().getCvFrame()

    # Resize the mono camera frames to match the color camera frame size
    left_frame = cv2.resize(left_frame, (640, 480))
    right_frame = cv2.resize(right_frame, (640, 480))

    # Preview the mono camera frames
    cv2.imshow('Left Camera', left_frame)
    cv2.imshow('Right Camera', right_frame)

    cv2.moveWindow('Left Camera', 0, 0)
    cv2.moveWindow('Right Camera', 1280, 0)

    # Exit the loop if 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
sentinel_lib.close_pipeline()

