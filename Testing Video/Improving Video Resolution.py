import cv2

def upscale_and_enhance(frame, scale_factor=1.3):
    """
    Upscales and enhances the frame.
    
    Args:
        frame: Input video frame.
        scale_factor: Factor to upscale the resolution.
    Returns:
        Enhanced and upscaled frame.
    """
    # Resize the frame (upsampling)
    height, width = frame.shape[:2]
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    upscaled_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    # sr = cv2.dnn_superres.DnnSuperResImpl_create()
    # sr.readModel("EDSR_x2.pb")
    # sr.setModel("edsr", 2)  # Use EDSR model with a scale of 2
    # enhanced_frame = sr.upsample(upscaled_frame)


    # Optional: Enhance contrast and brightness
    final_frame = cv2.convertScaleAbs(upscaled_frame, alpha=1.0, beta=0)

    return final_frame

def main():
    # Open the video capture (0 is usually the first USB/Analog camera)
    cap = cv2.VideoCapture(0)

    # cap.set(cv2.CAP_PROP_BRIGHTNESS, 50)  # Adjust brightness
    # cap.set(cv2.CAP_PROP_CONTRAST, 50)    # Adjust contrast
    # cap.set(cv2.CAP_PROP_EXPOSURE, -4)    # Adjust exposure

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open analog camera.")
        return

    print("Press 'q' to quit.")

    while True:
        # Capture a frame
        ret, frame = cap.read()

        if ret:
            # Enhance the frame resolution
            enhanced_frame = upscale_and_enhance(frame)

            # Display the enhanced frame
            cv2.imshow('Enhanced Analog Camera Feed', enhanced_frame)
        else:
            print("Error: Could not read frame.")
            break

        # Quit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close the window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
