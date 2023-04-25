import argparse
import cv2
import sentinel_lib

def main(args):
    # Initialize camera
    camera = sentinel_lib.init_camera(args.with_myriadx)

    # Main loop
    while True:
        # Get frames
        left_frame, right_frame, center_frame, nn_frame = sentinel_lib.get_frames(camera, args.with_myriadx)

        # Show preview windows
        cv2.imshow("Left Mono Camera", left_frame)
        cv2.imshow("Right Mono Camera", right_frame)
        cv2.imshow("Center Camera", center_frame)

        if args.with_myriadx:
            cv2.imshow("Neural Net Frame Differences", nn_frame)

        # Exit on key press
        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            break

    # Close camera
    sentinel_lib.close_camera(camera)

    # Close all windows
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
