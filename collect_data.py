import cv2
import os

# Define the path to save the images
output_folder = 'gesture_images'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Define the gesture labels
gestures = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# Function to collect images for each gesture
def collect_images():
    cap = cv2.VideoCapture(0)  # Start the webcam
    
    # Loop through each gesture
    for gesture in gestures:
        print(f"\nPlease show the gesture for {gesture} and press 'c' to capture.")
        print(f"Press 'q' to quit the collection process.")
        
        count = 0  # Counter for image saving
        
        while True:
            ret, frame = cap.read()  # Capture frame from webcam
            
            if not ret:
                print("Failed to grab frame")
                break
            
            # Display the captured frame
            cv2.putText(frame, f'Gesture: {gesture}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow('Capture Gesture', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c'):  # If 'c' is pressed, save the image
                image_path = os.path.join(output_folder, f'{gesture}_{count}.jpg')
                cv2.imwrite(image_path, frame)
                print(f"Saved image: {image_path}")
                count += 1
            
            if key == ord('q'):  # If 'q' is pressed, quit the process
                print("Exiting collection.")
                break
        
        # Wait for the user to press 'q' to move to the next gesture
        print(f"Finished collecting images for gesture {gesture}. Press 'q' to continue to the next gesture.")
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    cap.release()  # Release the webcam
    cv2.destroyAllWindows()  # Close all OpenCV windows

if __name__ == '__main__':
    collect_images()
