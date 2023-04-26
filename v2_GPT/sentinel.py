import cv2
import sentinel_lib

pipeline = sentinel_lib.create_pipeline()
sentinel_lib.start_pipeline(pipeline)

while True:
    color_frame = sentinel_lib.color_queue.get().getCvFrame()
    left_frame = sentinel_lib.left_queue.get().getCvFrame()
    right_frame = sentinel_lib.right_queue.get().getCvFrame()

    sentinel_lib.process_frame(color_frame, 'Color Camera')
    sentinel_lib.process_frame(left_frame, 'Left Mono Camera')
    sentinel_lib.process_frame(right_frame, 'Right Mono Camera')

    if cv2.waitKey(1) == ord('q'):
        break

sentinel_lib.close_pipeline()

