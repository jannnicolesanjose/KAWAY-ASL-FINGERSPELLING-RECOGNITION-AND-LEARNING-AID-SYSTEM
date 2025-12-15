from flask import Flask, render_template, Response, request
import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

app = Flask(__name__)

# Load the model
model_dict = pickle.load(open('../model/model.p', 'rb'))
model = model_dict['model']

cap = None

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Stabilization and word-building variables
recent_predictions = deque(maxlen=5)  # Buffer for recent predictions
stable_letter = None
last_letter = None  # Initialize last_letter here
last_letter_time = 0
letter_cooldown = 3.0  # Cooldown time in seconds between same letters
spelled_word = ""  # Word being built

# Functions for gesture recognition (optional)
def is_open_palm(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    return thumb_tip.x < index_tip.x < middle_tip.x < pinky_tip.x  # Spread fingers

def is_fist(hand_landmarks):
    for landmark in hand_landmarks.landmark:
        if landmark.y < hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y:
            return False
    return True

# Timing-based prediction stabilization variables
current_stable_start_time = None  # Time when the current stable letter detection started
stable_duration_required = 0.5  # Time in seconds the letter must be held stable

# Add a global variable to store the latest inference time
latest_inference_time = None


# Function to initialize the camera if not already initialized
def initialize_camera():
    global cap
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)  # Re-initialize the camera if not already open


# Video streaming generator function
def generate_frames():
    global stable_letter, spelled_word, last_letter, last_letter_time, current_stable_start_time, latest_inference_time

    # Ensure the camera is initialized each time the feed is requested
    initialize_camera()

    while True:
        data_aux = []
        x_ = []
        y_ = []

        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)

        # Always draw the "Word" label continuously
        brightness = np.mean(frame)  # Simple way to detect the background brightness
        text_color = (255, 255, 255) if brightness < 127 else (0, 0, 0)  # White or black text based on brightness
        outline_color = (0, 0, 0) if text_color == (255, 255, 255) else (255, 255, 255)

        # Always display the word at the top-left corner
        cv2.putText(frame, f"Word: {spelled_word}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, outline_color, 4, cv2.LINE_AA)  # Outline
        cv2.putText(frame, f"Word: {spelled_word}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, text_color, 2, cv2.LINE_AA)  # Main text

        if results.multi_hand_landmarks:
            # If hand is detected, process the landmarks
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

                # Prepare data for prediction
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            if len(data_aux) == 84:
                print(f"Skipping frame with {len(data_aux)} features (84).")
                continue

            # Predict with the model
            start_time = time.time()
            prediction = model.predict([np.asarray(data_aux)])
            confidence = np.max(model.predict_proba([np.asarray(data_aux)]))  # Extract confidence score
            end_time = time.time()
            predicted_character = prediction[0]

            # Update the global inference time
            latest_inference_time = end_time - start_time

            # Stabilize predictions
            recent_predictions.append(predicted_character)

            if len(set(recent_predictions)) == 1:  # If all predictions in the buffer are the same
                detected_letter = recent_predictions[0]

                if stable_letter != detected_letter:
                    # Start timing for the new letter
                    stable_letter = detected_letter
                    current_stable_start_time = time.time()
                else:
                    # Check how long the letter has been stable
                    elapsed_time = time.time() - current_stable_start_time

                    if elapsed_time >= stable_duration_required:
                        # Register the letter as valid and add it to the spelled word
                        if stable_letter != last_letter or (time.time() - last_letter_time > letter_cooldown):
                            last_letter = stable_letter
                            last_letter_time = time.time()
                            spelled_word += stable_letter
                            print(f"Spelled Word: {spelled_word}")
                            print(f"Inference Time: {latest_inference_time:.2f} s")


            # Calculate the bounding box for the hand
            if x_ and y_:
                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box for hand

                # Add text at bounding box position with confidence score
                cv2.putText(frame, f"{stable_letter if stable_letter else ''}    [{confidence:.2f}]",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, outline_color, 5, cv2.LINE_AA)   # Outline
                cv2.putText(frame, f"{stable_letter if stable_letter else ''}    [{confidence:.2f}]",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, text_color, 3, cv2.LINE_AA)  # Main text


        # Encode the frame to send it over HTTP
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print("Failed to encode frame.")
            break

        frame_data = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n\r\n')


# Route to serve the video stream
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to render the main page
@app.route('/')
def home():
    return render_template('home.html')  # This renders the index.html from the templates folder

@app.route('/Learn and About Us')
def learn_and_about_us():
    return render_template('Learn and About Us.html')  # New page in templates folder

@app.route('/learn')
def learn():
    return render_template('Learn.html')  # New page in templates folder

@app.route('/practice')
def practice():
    return render_template('Practice.html')  # Link to Practice page

@app.route('/easy')
def easy():
    return render_template('Easy.html')  # Link to Practice page

@app.route('/medium')
def medium():
    return render_template('Medium.html')  # Link to Practice page

@app.route('/hard')
def hard():
    return render_template('Hard.html')  # Link to Practice page

@app.route('/translate')
def translate():
    return render_template('Translate.html')  # Link to Translate page

@app.route('/asl_to_text')
def translate_asl_to_text():
    return render_template('ASLtoText.html')  # Adjust the template name accordingly

@app.route('/a')
def a():
    return render_template('LessonLetterA.html')  # Adjust the template name accordingly

@app.route('/b')
def b():
    return render_template('LessonLetterB.html')  # Adjust the template name accordingly

@app.route('/c')
def c():
    return render_template('LessonLetterC.html')  # Adjust the template name accordingly

@app.route('/d')
def d():
    return render_template('LessonLetterD.html')  # Adjust the template name accordingly

@app.route('/e')
def e():
    return render_template('LessonLetterE.html')  # Adjust the template name accordingly

@app.route('/f')
def f():
    return render_template('LessonLetterF.html')  # Adjust the template name accordingly

@app.route('/g')
def g():
    return render_template('LessonLetterG.html')  # Adjust the template name accordingly

@app.route('/h')
def h():
    return render_template('LessonLetterH.html')  # Adjust the template name accordingly

@app.route('/i')
def i():
    return render_template('LessonLetterI.html')  # Adjust the template name accordingly

@app.route('/j')
def j():
    return render_template('LessonLetterJ.html')  # Adjust the template name accordingly

@app.route('/k')
def k():
    return render_template('LessonLetterK.html')  # Adjust the template name accordingly

@app.route('/l')
def l():
    return render_template('LessonLetterL.html')  # Adjust the template name accordingly

@app.route('/m')
def m():
    return render_template('LessonLetterM.html')  # Adjust the template name accordingly

@app.route('/n')
def n():
    return render_template('LessonLetterN.html')  # Adjust the template name accordingly

@app.route('/o')
def o():
    return render_template('LessonLetterO.html')  # Adjust the template name accordingly

@app.route('/p')
def p():
    return render_template('LessonLetterP.html')  # Adjust the template name accordingly

@app.route('/q')
def q():
    return render_template('LessonLetterQ.html')  # Adjust the template name accordingly

@app.route('/r')
def r():
    return render_template('LessonLetterR.html')  # Adjust the template name accordingly

@app.route('/s')
def s():
    return render_template('LessonLetterS.html')  # Adjust the template name accordingly

@app.route('/t')
def t():
    return render_template('LessonLetterT.html')  # Adjust the template name accordingly

@app.route('/u')
def u():
    return render_template('LessonLetterU.html')  # Adjust the template name accordingly

@app.route('/v')
def v():
    return render_template('LessonLetterV.html')  # Adjust the template name accordingly

@app.route('/w')
def w():
    return render_template('LessonLetterW.html')  # Adjust the template name accordingly

@app.route('/x')
def x():
    return render_template('LessonLetterX.html')  # Adjust the template name accordingly

@app.route('/y')
def y():
    return render_template('LessonLetterY.html')  # Adjust the template name accordingly

@app.route('/z')
def z():
    return render_template('LessonLetterZ.html')  # Adjust the template name accordingly


@app.route('/text_to_asl')
def translate_text_to_asl():
    return render_template('TexttoASL.html')  # Adjust the template name accordingly

@app.route('/about us')
def about_us():
    return render_template('AboutUs.html')  # Link to About Us page

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global cap, recent_predictions, stable_letter, last_letter, last_letter_time, spelled_word, current_stable_start_time

    # Stop the camera if it's running
    if cap and cap.isOpened():
        cap.release()
        print("Camera stopped.")

    # Clear the prediction and state variables
    recent_predictions.clear()  # Clear the prediction buffer
    stable_letter = None  # Reset the current stable letter
    last_letter = None  # Reset the last letter detected
    last_letter_time = 0  # Reset the time for the last letter
    spelled_word = ""  # Reset the spelled word
    current_stable_start_time = None  # Reset the time for stable prediction

    return '', 204  # No Content response


@app.route('/delete_last_character', methods=['POST'])
def delete_last_character():
    global spelled_word
    if spelled_word:
        spelled_word = spelled_word[:-1]
    return '', 204


@app.route('/get_predicted_word')
def get_predicted_word():
    return {'predicted_word': spelled_word}


@app.route('/get_inference_time')
def get_inference_time():
    return {'inference_time': f"{latest_inference_time:.2f} sec" if latest_inference_time is not None else ""}


if __name__ == '__main__':
    app.run(debug=True)
