import cv2
import mediapipe as mp
import random
import pygame

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                      max_num_hands=1,
                      min_detection_confidence=0.7,
                      min_tracking_confidence=0.7)

mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
choices = ['paper', 'scissors', 'rock']
computer_choice = random.choice(choices)
player_score = 0
computer_score = 0
round_count = 0
gesture = ""
score_updated = False
thumb_up_detected = False

def is_thumb_up(fingers_open):
    return fingers_open[0] and not fingers_open[1] and not fingers_open[2] and not fingers_open[3]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks: 
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            finger_tips = [hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],
                           hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
                           hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP],
                           hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]]

            finger_pips = [hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP],
                          hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP],
                          hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP],
                          hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]]

            fingers_open = []
            for tip, pip in zip(finger_tips, finger_pips):
                if tip.y < pip.y:
                    fingers_open.append(True)
                else:
                    fingers_open.append(False)

            if all(fingers_open):
                gesture = "paper"
            elif fingers_open[0] and fingers_open[1] and not fingers_open[2] and not fingers_open[3]:
                gesture = "scissors"
            else:
                gesture = "rock"

            if is_thumb_up(fingers_open) and not thumb_up_detected:
                thumb_up_detected = True
            elif not is_thumb_up(fingers_open) and thumb_up_detected:
                round_count += 1
                computer_choice = random.choice(choices)
                score_updated = False
                thumb_up_detected = False
                
                pygame.mixer.init()
                pygame.mixer.music.load("round.mp3")
                pygame.mixer.music.play()
            
            if gesture and not score_updated:
                if gesture == computer_choice:
                    cv2.putText(frame, "It's a tie!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                elif gesture == 'scissors' and computer_choice == 'rock':
                    computer_score += 1
                    cv2.putText(frame, "You lose!", (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2 )
                    pygame.mixer.init()
                    pygame.mixer.music.load("loss.mp3")
                    pygame.mixer.music.play()
                    
                elif gesture == 'paper' and computer_choice == 'rock':
                    player_score += 1
                    cv2.putText(frame, "You win!", (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    pygame.mixer.init()
                    pygame.mixer.music.load("win.mp3")
                    pygame.mixer.music.play()
                    
                elif gesture == 'rock' and computer_choice == 'scissors':
                    player_score += 1
                    cv2.putText(frame, "You win!", (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    pygame.mixer.init()
                    pygame.mixer.music.load("win.mp3")
                    pygame.mixer.music.play()
                    
                elif gesture == 'rock' and computer_choice == 'paper':
                    computer_score += 1
                    cv2.putText(frame, "You lose!", (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    pygame.mixer.init()
                    pygame.mixer.music.load("loss.mp3")
                    pygame.mixer.music.play()
                    
                elif gesture == 'scissors' and computer_choice == 'paper':
                    player_score += 1
                    cv2.putText(frame, "You win!", (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    pygame.mixer.init()
                    pygame.mixer.music.load("win.mp3")
                    pygame.mixer.music.play()
                    
                elif gesture == 'paper' and computer_choice == 'scissors':
                    computer_score += 1
                    cv2.putText(frame, "You lose!", (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    pygame.mixer.init()
                    pygame.mixer.music.load("loss.mp3")
                    pygame.mixer.music.play()
                score_updated = True

            cv2.putText(frame, f"Gesture: {gesture}", (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, f"Computer: {computer_choice}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, f"Round: {round_count}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.putText(frame, f"Player Score: {player_score}", (400, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.putText(frame, f"Computer Score: {computer_score}", (400, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    cv2.imshow('Hand Gesture Detection', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
