import time
import cv2
import depthai as dai
import apriltag
import sentinel_lib as s_lib
import dynamixel_lib
import threading
import os

stop_preview_thread = False

def move_servos_to_tag(tag_id, pan_position, tilt_position):
    dynamixel_lib.move_servo(dynamixel_lib.DXL_ID_PAN, pan_position)
    dynamixel_lib.move_servo(dynamixel_lib.DXL_ID_TILT, tilt_position)
    time.sleep(1)

def show_preview():
    global stop_preview_thread
    while not stop_preview_thread:
        color_frame = s_lib.color_queue.get().getCvFrame()
        left_frame = s_lib.left_queue.get().getCvFrame()
        right_frame = s_lib.right_queue.get().getCvFrame()

        # Draw a neon green dot in the center of each camera view
        dot_color = (173, 255, 47)
        dot_radius = 5
        center = (color_frame.shape[1] // 2, color_frame.shape[0] // 2)
        cv2.circle(color_frame, center, dot_radius, dot_color, -1)
        cv2.circle(left_frame, center, dot_radius, dot_color, -1)
        cv2.circle(right_frame, center, dot_radius, dot_color, -1)

        cv2.imshow('Color Camera', color_frame)
        cv2.imshow('Left Mono Camera', left_frame)
        cv2.imshow('Right Mono Camera', right_frame)

        key = cv2.waitKey(1)
        if key == ord("q") or key == ord("Q"):
            break

def main():
    # Load the calibration data if it exists
    calibration_data_file = "calibration_data.json"
    calibration_data = None
    if os.path.exists(calibration_data_file):
        calibration_data = s_lib.load_calibration_data(calibration_data_file)

    # Create and start the pipeline
    pipeline = s_lib.create_pipeline()
    s_lib.start_pipeline(pipeline)

    # Create a thread to show the camera previews
    preview_thread = threading.Thread(target=show_preview)
    preview_thread.daemon = True
    preview_thread.start()

    # Enable torque on the servos
    print("Enabling torque on the servos...")
    dynamixel_lib.set_servo_torque(dynamixel_lib.DXL_ID_PAN, True)
    dynamixel_lib.set_servo_torque(dynamixel_lib.DXL_ID_TILT, True)

    if calibration_data:
        print("Calibration data found. Moving servos to tag positions.")
        for tag_id, position in calibration_data['tag_positions'].items():
            print(f"Moving to tag {tag_id} position...")
            move_servos_to_tag(tag_id, position[0], position[1])
            input(f"Servos are at tag {tag_id} position. Press Enter to continue.")

    # Proceed with the calibration process
    print("Starting the calibration process...")
    # Add your calibration process code here

    s_lib.close_pipeline()

    # Signal the preview thread to stop
    global stop_preview_thread
    stop_preview_thread = True
    preview_thread.join()

if __name__ == "__main__":
    main()

