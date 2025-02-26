#Face and Hand Detection
import cv2
import mediapipe as mp

#for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
total_face = 0
# initialize hands for hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Open the camera
cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if not ret:
        break

    # Convert the frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
    total_face += len(faces) 
    
    # Draw rectangles around face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Convert the frame to RGB for hand detection
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Draw hand lines 
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                h, w, _ = frame.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1) 
            
            # draw connection line between points
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    #display
    cv2.imshow('Face and Hand Detection', frame)

    if cv2.waitKey(20) & 0xFF == 27:
        break

#close all the windows
print(f"total faces are : {total_face}")
cam.release()
cv2.destroyAllWindows()














# import cv2
# import mediapipe as mp
# import random

# # Initialize MediaPipe Hands
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# # Initialize drawing styles
# mp_drawing = mp.solutions.drawing_utils

# # Initialize the camera
# cap = cv2.VideoCapture(0)

# # Initialize scores
# user_score = 0
# computer_score = 0

# while cap.isOpened():
#     success, image = cap.read()
#     if not success:
#         print("Ignoring empty camera frame.")
#         continue

#     # Convert the BGR image to RGB.
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
#     # Process the RGB image
#     results = hands.process(image)

#     # Draw the hand annotations on the image.
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
#     user_gesture = "Unknown"
    
#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             mp_drawing.draw_landmarks(
#                 image,
#                 hand_landmarks,
#                 mp_hands.HAND_CONNECTIONS,
#                 mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
#                 mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
#             )
            
#             # Determine finger states
#             thumb_tip = hand_landmarks.landmark[4].y
#             thumb_mcp = hand_landmarks.landmark[2].y
#             thumb_ip = hand_landmarks.landmark[3].y

#             index_tip = hand_landmarks.landmark[8].y
#             index_mcp = hand_landmarks.landmark[5].y

#             middle_tip = hand_landmarks.landmark[12].y
#             middle_mcp = hand_landmarks.landmark[9].y

#             ring_tip = hand_landmarks.landmark[16].y
#             ring_mcp = hand_landmarks.landmark[13].y

#             pinky_tip = hand_landmarks.landmark[20].y
#             pinky_mcp = hand_landmarks.landmark[17].y

#             thumb_open = thumb_tip < thumb_mcp
#             index_open = index_tip < index_mcp
#             middle_open = middle_tip < middle_mcp
#             ring_open = ring_tip < ring_mcp
#             pinky_open = pinky_tip < pinky_mcp

#             # Classify gestures
#             if not (thumb_open or index_open or middle_open or ring_open or pinky_open):
#                 user_gesture = "Rock"
#             elif thumb_open and index_open and middle_open and ring_open and pinky_open:
#                 user_gesture = "Paper"
#             elif index_open and middle_open and not (thumb_open or ring_open or pinky_open):
#                 user_gesture = "Scissors"
    
#     # Generate computer's gesture
#     computer_gesture = random.choice(["Rock", "Paper", "Scissors"])
    
#     # Determine the winner
#     if user_gesture == computer_gesture:
#         winner = "Tie"
#     elif (user_gesture == "Rock" and computer_gesture == "Scissors") or \
#          (user_gesture == "Scissors" and computer_gesture == "Paper") or \
#          (user_gesture == "Paper" and computer_gesture == "Rock"):
#         winner = "User"
#         user_score += 1
#     else:
#         winner = "Computer"
#         computer_score += 1
    
#     # Display results
#     cv2.putText(image, f"User Gesture: {user_gesture}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
#     cv2.putText(image, f"Computer Gesture: {computer_gesture}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
#     cv2.putText(image, f"Winner: {winner}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
#     cv2.putText(image, f"User Score: {user_score}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
#     cv2.putText(image, f"Computer Score: {computer_score}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
#     # Display the resulting image
#     cv2.imshow('MediaPipe Hand Tracking', image)
    
#     if cv2.waitKey(5) & 0xFF == 27:
#         break

# hands.close()
# cap.release()
# cv2.destroyAllWindows()
