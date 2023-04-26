import argparse
import cv2
import sentinel_lib
import numpy as np

def main(args):
    camera = sentinel_lib.init_camera(args.with_myriadx)

    while True:
        left_frame, right_frame, center_frame, nn_frame = sentinel_lib.get_frames(camera, args.with_myriadx)

        cv2.imshow("Left Camera", left_frame)
        cv2.imshow("Right Camera", right_frame)
        cv2.imshow("Center Camera", center_frame)

        if args.with_myriadx:
            nn_frame = np.array(nn_frame).reshape(720, 720)  # Reshape the array to match the image dimensions
            nn_frame_normalized = cv2.normalize(nn_frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            cv2.imshow("Neural Net Frame Differences", nn_frame_normalized)

        key = cv2.waitKey(1)
        if key == 27:  # ESC key
            break

    sentinel_lib.close_camera(camera)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-myriadx", action="store_true", help="Use MyriadX support")
    parser.add_argument("--without-myriadx", action="store_true", help="Don't use MyriadX support")
    args = parser.parse_args()

    # Check if both arguments are provided
    if args.with_myriadx and args.without_myriadx:
        raise ValueError("Cannot use both --with-myriadx and --without-myriadx flags")

    main(args)
