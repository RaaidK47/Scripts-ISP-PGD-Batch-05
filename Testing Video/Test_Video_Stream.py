import cv2

def main():
    # Open the video capture (0 is usually the first USB camera)
    cap = cv2.VideoCapture(1)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open USB camera.")
        return

    print("Press 'q' to quit.")

    while True:
        # Capture a frame
        ret, frame = cap.read()

        # If frame capture was successful
        if ret:
            # Display the frame
            cv2.imshow('USB Camera Feed', frame)
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
