import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model

# Load your trained model
model = load_model('gesture_model.h5')  # <-- update path if needed

# Print model input size
input_shape = model.input_shape
_, height, width, channels = input_shape
print(f"Model expects input shape: {height}x{width}x{channels}")

# Load class names
class_names = ['1', '2', '3', '4', '5']  # <-- replace with your actual class names

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Drawing utility
mp_drawing = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame (mirror image)
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process with MediaPipe
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get bounding box from landmarks
            img_height, img_width, _ = frame.shape
            landmark_array = np.array(
                [(lm.x * img_width, lm.y * img_height) for lm in hand_landmarks.landmark]
            )
            x_min = int(np.min(landmark_array[:, 0]) - 20)
            y_min = int(np.min(landmark_array[:, 1]) - 20)
            x_max = int(np.max(landmark_array[:, 0]) + 20)
            y_max = int(np.max(landmark_array[:, 1]) + 20)

            # Ensure the box stays within the frame
            x_min = max(x_min, 0)
            y_min = max(y_min, 0)
            x_max = min(x_max, img_width)
            y_max = min(y_max, img_height)

            # Crop hand region
            hand_img = frame[y_min:y_max, x_min:x_max]
            if hand_img.size == 0:
                continue

            # Preprocess the cropped hand for prediction
            hand_img = cv2.resize(hand_img, (width, height))  # Auto uses model's expected width and height
            hand_img = hand_img / 255.0
            hand_img = np.expand_dims(hand_img, axis=0)

            # Predict gesture
            prediction = model.predict(hand_img)  # <-- Added model prediction
            print(f"Prediction shape: {prediction.shape}")
            predicted_class = class_names[np.argmax(prediction)]  # Get the predicted class

            # Draw the bounding box and prediction
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, predicted_class, (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Draw hand landmarks (optional)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Hand Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
