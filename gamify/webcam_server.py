from flask import Flask, render_template, Response
import cv2
import pyautogui
from time import time
from math import hypot
import mediapipe as mp
import matplotlib.pyplot as plt

app = Flask(__name__)
mp_pose = mp.solutions.pose
pose_video = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Constants for calorie count (approximations)
CALORIES_PER_JUMP = 0.2
CALORIES_PER_CROUCH = 0.15

# Initialize jump and crouch counts
jump_count = 0
crouch_count = 0
calories_burnt = 0

# Initialize the previous posture
previous_posture = 'Standing'

def detectPose(image, pose, draw=False):
    output_image = image.copy()
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(imageRGB)

    if results.pose_landmarks and draw:
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=3, circle_radius=3),
                                  connection_drawing_spec=mp_drawing.DrawingSpec(color=(49, 125, 237), thickness=2, circle_radius=2))

    return output_image, results

def checkHandsJoined(image, results, draw=False):
    height, width, _ = image.shape
    output_image = image.copy()

    left_wrist_landmark = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * width,
                           results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * height)
    right_wrist_landmark = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x * width,
                            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * height)
    euclidean_distance = int(hypot(left_wrist_landmark[0] - right_wrist_landmark[0],
                                   left_wrist_landmark[1] - right_wrist_landmark[1]))

    if euclidean_distance < 130:
        hand_status = 'Hands Joined'
        color = (0, 255, 0)
    else:
        hand_status = 'Hands Not Joined'
        color = (0, 0, 255)

    if draw:
        cv2.putText(output_image, hand_status, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 3)
        cv2.putText(output_image, f'Distance: {euclidean_distance}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, color, 3)

    return output_image, hand_status

def checkLeftRight(image, results, draw=False):
    horizontal_position = None
    height, width, _ = image.shape
    output_image = image.copy()

    left_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * width)
    right_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * width)

    if (right_x <= width // 2 and left_x <= width // 2):
        horizontal_position = 'Left'
    elif (right_x >= width // 2 and left_x >= width // 2):
        horizontal_position = 'Right'
    elif (right_x >= width // 2 and left_x <= width // 2):
        horizontal_position = 'Center'

    if draw:
        cv2.putText(output_image, horizontal_position, (5, height - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)
        cv2.line(output_image, (width // 2, 0), (width // 2, height), (255, 255, 255), 2)

    return output_image, horizontal_position

def checkJumpCrouch(image, results, MID_Y=250, draw=False):
    global jump_count, crouch_count, calories_burnt, previous_posture
    height, width, _ = image.shape
    output_image = image.copy()

    left_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * height)
    right_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * height)
    actual_mid_y = abs(right_y + left_y) // 2

    lower_bound = MID_Y - 15
    upper_bound = MID_Y + 100

    if actual_mid_y < lower_bound:
        current_posture = 'Jumping'
    elif actual_mid_y > upper_bound:
        current_posture = 'Crouching'
    else:
        current_posture = 'Standing'

    # Only increment counts when transitioning from "Standing" to either "Jumping" or "Crouching"
    if previous_posture == 'Standing' and current_posture == 'Jumping':
        jump_count += 1
        calories_burnt += CALORIES_PER_JUMP
    elif previous_posture == 'Standing' and current_posture == 'Crouching':
        crouch_count += 1
        calories_burnt += CALORIES_PER_CROUCH

    # Update the previous posture
    previous_posture = current_posture

    if draw:
        cv2.putText(output_image, current_posture, (5, height - 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)
        cv2.line(output_image, (0, MID_Y), (width, MID_Y), (255, 255, 255), 2)

    return output_image, current_posture

def generate_frames():
    global jump_count, crouch_count, calories_burnt
    camera_video = cv2.VideoCapture(0)
    camera_video.set(3, 1280)
    camera_video.set(4, 960)

    time1 = 0
    game_started = False
    x_pos_index = 1
    y_pos_index = 1
    MID_Y = None
    counter = 0
    num_of_frames = 10

    while camera_video.isOpened():
        ok, frame = camera_video.read()
        if not ok:
            continue

        frame = cv2.flip(frame, 1)
        frame, results = detectPose(frame, pose_video, draw=game_started)

        if results.pose_landmarks:
            if game_started:
                frame, horizontal_position = checkLeftRight(frame, results, draw=True)
                if (horizontal_position == 'Left' and x_pos_index != 0) or (horizontal_position == 'Center' and x_pos_index == 2):
                    pyautogui.press('left')
                    x_pos_index -= 1
                elif (horizontal_position == 'Right' and x_pos_index != 2) or (horizontal_position == 'Center' and x_pos_index == 0):
                    pyautogui.press('right')
                    x_pos_index += 1
            else:
                cv2.putText(frame, 'JOIN BOTH HANDS TO START THE GAME.', (5, frame.shape[0] - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)

            if checkHandsJoined(frame, results)[1] == 'Hands Joined':
                counter += 1
                if counter == num_of_frames:
                    if not game_started:
                        game_started = True
                        left_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * frame.shape[0])
                        right_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * frame.shape[0])
                        MID_Y = abs(right_y + left_y) // 2
                        pyautogui.click(x=1300, y=800, button='left')
                    else:
                        pyautogui.press('space')
                    counter = 0
            else:
                counter = 0

            if MID_Y:
                frame, posture = checkJumpCrouch(frame, results, MID_Y, draw=True)
                if posture == 'Jumping' and y_pos_index == 1:
                    pyautogui.press('up')
                    y_pos_index += 1
                elif posture == 'Crouching' and y_pos_index == 1:
                    pyautogui.press('down')
                    y_pos_index -= 1
                elif posture == 'Standing' and y_pos_index != 1:
                    y_pos_index = 1

        else:
            counter = 0

        # Display FPS
        time2 = time()
        if (time2 - time1) > 0:
            frames_per_second = 1.0 / (time2 - time1)
            cv2.putText(frame, 'FPS: {}'.format(int(frames_per_second)), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
        time1 = time2

        cv2.putText(frame, f'Jumps: {jump_count}', (frame.shape[1] - 300, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
        cv2.putText(frame, f'Crouches: {crouch_count}', (frame.shape[1] - 300, 60), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
        cv2.putText(frame, f'Calories: {calories_burnt:.2f}', (frame.shape[1] - 300, 90), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    camera_video.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
