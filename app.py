import cv2
import os
import shutil
from flask import Flask, Response, request, render_template, jsonify
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
import requests
import base64
import threading

#### Defining Flask App
app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

#### API configuration (for Docker networking)
API_HOST = os.environ.get('API_HOST', 'localhost')
API_PORT = os.environ.get('API_PORT', '3000')
API_PROTOCOL = 'https' if API_PORT == '443' else 'http'


#### Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%A, %B %d, %Y")


#### Initialize face detector (no camera - browser handles that)
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#### Global variables for browser-based camera
latest_frame = None
frame_lock = threading.Lock()
attendance_mode = False
add_user_mode = False
add_user_id = None
add_user_count = 0
detected_person = None
detection_start_time = None
last_face_seen_time = None  # Track when we last saw a face


#### If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv','w') as f:
        f.write('Name,Roll,Time')


#### get a number of total registered users
def totalreg():
    return len(os.listdir('static/faces'))


#### extract the face from an image
def extract_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if face_detector.empty():
        print("Error: Haarcascade file not loaded properly.")
        return []
    face_points = face_detector.detectMultiScale(gray, 1.3, 5)
    return face_points if len(face_points) > 0 else []



#### Identify face using ML model
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)


#### Identify face with confidence score
def identify_face_with_confidence(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    # Get distances to nearest neighbors
    distances, indices = model.kneighbors(facearray, n_neighbors=1)
    prediction = model.predict(facearray)
    # Return prediction and average distance (lower = more confident)
    return prediction[0], distances[0][0]


# Threshold for unknown face detection (adjust based on testing)
# Lower = stricter (rejects more), Higher = lenient (accepts more)
UNKNOWN_THRESHOLD = int(os.environ.get('UNKNOWN_THRESHOLD', '4500'))  # If distance > this, face is unknown


#### A function which trains the model on all the faces available in faces folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces,labels)
    joblib.dump(knn,'static/face_recognition_model.pkl')


#### Extract info from today's attendance file in attendance folder
def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names,rolls,times,l


#### Add Attendance of a specific user
def add_attendance(name):
    userid = name
    current_time = datetime.now().strftime("%H:%M:%S")


#### Decode base64 image from browser
def decode_frame(data_url):
    try:
        header, encoded = data_url.split(',', 1)
        img_data = base64.b64decode(encoded)
        np_arr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        print(f"Error decoding frame: {e}")
        return None


################## ROUTING FUNCTIONS #########################

import time

#### Our main page
@app.route('/')
def home():
    names, rolls, times, l = extract_attendance()
    register_param = request.args.get('register', '')
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, register=register_param)


#### Start attendance mode (returns JSON for browser-based flow)
@app.route('/start', methods=['GET'])
def start():
    global attendance_mode, detected_person, detection_start_time

    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return jsonify({'error': 'No trained model found. Please add a new face first.'}), 400

    attendance_mode = True
    detected_person = None
    detection_start_time = None

    return jsonify({'status': 'started', 'message': 'Attendance mode started. Look at the camera.'})


#### Stop attendance mode
@app.route('/stop', methods=['GET'])
def stop():
    global attendance_mode, add_user_mode
    attendance_mode = False
    add_user_mode = False
    return jsonify({'status': 'stopped'})


#### Process frame from browser for attendance
@app.route('/process_frame', methods=['POST'])
def process_frame():
    global detected_person, detection_start_time, attendance_mode, last_face_seen_time

    if not attendance_mode:
        return jsonify({'status': 'idle'})

    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image data'}), 400

    frame = decode_frame(data['image'])
    if frame is None:
        return jsonify({'error': 'Failed to decode image'}), 400

    faces = extract_faces(frame)

    if len(faces) == 0:
        # If scan was in progress (timer started) and face disappears, halt immediately
        if detection_start_time is not None:
            attendance_mode = False
            detected_person = None
            detection_start_time = None
            last_face_seen_time = None
            return jsonify({
                'status': 'face_lost',
                'message': 'Scanning halted: Face left the frame. Please try again.'
            })
        # Only reset idle state if no face detected for more than 1.5 seconds
        if last_face_seen_time and (time.time() - last_face_seen_time) > 1.5:
            detected_person = None
            last_face_seen_time = None
        return jsonify({'status': 'no_face', 'message': 'No face detected'})

    # Update last face seen time
    last_face_seen_time = time.time()

    (x, y, w, h) = faces[0]
    face = cv2.resize(frame[y:y + h, x:x + w], (50, 50))

    try:
        identified_person, distance = identify_face_with_confidence(face.reshape(1, -1))
        print(f"[DEBUG] Detected: {identified_person}, Distance: {distance}")  # Check Railway logs
    except Exception as e:
        return jsonify({'error': f'Recognition error: {str(e)}'}), 500

    # Check if face is unknown (distance too high)
    is_unknown = distance > UNKNOWN_THRESHOLD

    if is_unknown or detected_person != identified_person:
        # If a scan was already in progress (timer started) and person changed or unknown appeared, halt the scan
        if detection_start_time is not None:
            attendance_mode = False
            detected_person = None
            detection_start_time = None
            last_face_seen_time = None
            message = 'Scanning halted: Unregistered person detected. Please try again.' if is_unknown else 'Scanning halted: Different person detected. Please try again.'
            return jsonify({
                'status': 'person_changed',
                'message': message
            })

        # If unknown person tries to start a scan, reject immediately
        if is_unknown:
            return jsonify({
                'status': 'unknown',
                'message': 'Unregistered person. Please register first.'
            })

        # First detection of a person, start the timer
        detected_person = identified_person
        detection_start_time = time.time()
        return jsonify({
            'status': 'detecting',
            'person': identified_person,
            'message': f'Detected: {identified_person}. Hold still for 5 seconds...',
            'elapsed': 0
        })

    elapsed = time.time() - detection_start_time if detection_start_time else 0

    if elapsed >= 5:
        # Attendance confirmed - send API request
        try:
            api_url = f'{API_PROTOCOL}://{API_HOST}:{API_PORT}/attendances'
            api_response = requests.post(
                api_url,
                json={'attendances': {'customer_number': identified_person}},
                timeout=10
            )
            print(f"API Response ({api_url}): {api_response.status_code}")
        except Exception as e:
            print(f"API Error: {e}")

        attendance_mode = False
        detected_person = None
        detection_start_time = None
        last_face_seen_time = None

        return jsonify({
            'status': 'completed',
            'person': identified_person,
            'message': f'Attendance recorded for {identified_person}!'
        })

    return jsonify({
        'status': 'detecting',
        'person': identified_person,
        'message': f'Detected: {identified_person}. Hold still... ({int(5 - elapsed)}s)',
        'elapsed': elapsed
    })


#### Start add user mode
@app.route('/add', methods=['POST'])
def add():
    global add_user_mode, add_user_id, add_user_count

    newuserid = request.form.get('newuserid')
    if not newuserid:
        return jsonify({'error': 'User ID is required'}), 400

    userimagefolder = f'static/faces/{newuserid}'
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)

    add_user_mode = True
    add_user_id = newuserid
    add_user_count = 0

    return jsonify({
        'status': 'started',
        'user_id': newuserid,
        'message': f'Started capturing faces for {newuserid}. Look at the camera.'
    })


#### Process frame for adding new user
@app.route('/add_frame', methods=['POST'])
def add_frame():
    global add_user_mode, add_user_id, add_user_count

    if not add_user_mode or not add_user_id:
        return jsonify({'status': 'idle'})

    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image data'}), 400

    frame = decode_frame(data['image'])
    if frame is None:
        return jsonify({'error': 'Failed to decode image'}), 400

    faces = extract_faces(frame)

    if len(faces) == 0:
        return jsonify({
            'status': 'capturing',
            'count': add_user_count,
            'message': 'No face detected. Please look at the camera.'
        })

    (x, y, w, h) = faces[0]
    face_crop = frame[y:y + h, x:x + w]

    # Save face image
    userimagefolder = f'static/faces/{add_user_id}'
    img_name = f"{add_user_id}_{add_user_count}.jpg"
    img_path = os.path.join(userimagefolder, img_name)
    cv2.imwrite(img_path, face_crop)
    add_user_count += 1

    if add_user_count >= 50:
        # Done capturing, train the model
        print(f"Images stored in {userimagefolder}")
        print("Training Model with New Data...")
        train_model()

        add_user_mode = False
        saved_id = add_user_id
        add_user_id = None
        add_user_count = 0

        # Send face scan update to API
        try:
            api_url = f'{API_PROTOCOL}://{API_HOST}:{API_PORT}/members/update_face_scan'
            api_response = requests.post(
                api_url,
                json={'customer_number': saved_id},
                timeout=10
            )
            print(f"API Response ({api_url}): {api_response.status_code}")
            if api_response.status_code == 200:
                print(f"Successfully updated face scan for {saved_id}")
            else:
                print(f"API Response Body: {api_response.text}")
        except Exception as e:
            print(f"API Error: {e}")

        return jsonify({
            'status': 'completed',
            'user_id': saved_id,
            'message': f'Successfully added {saved_id}! Model trained.'
        })

    return jsonify({
        'status': 'capturing',
        'count': add_user_count,
        'message': f'Capturing images: {add_user_count}/50'
    })


#### Get current status
@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        'attendance_mode': attendance_mode,
        'add_user_mode': add_user_mode,
        'add_user_id': add_user_id,
        'add_user_count': add_user_count
    })


#### Unregister a face from the model
@app.route('/unregister/<user_id>', methods=['DELETE'])
def unregister(user_id):
    userimagefolder = f'static/faces/{user_id}'

    # Check if user exists
    if not os.path.isdir(userimagefolder):
        return jsonify({'error': f'User {user_id} not found'}), 404

    try:
        # Delete user's face images directory
        shutil.rmtree(userimagefolder)
        print(f"Deleted face images for {user_id}")

        # Check if there are remaining users
        remaining_users = os.listdir('static/faces')

        if len(remaining_users) > 0:
            # Retrain model without the deleted user
            print("Retraining model without deleted user...")
            train_model()
            print("Model retrained successfully")
        else:
            # No users left, delete the model file
            model_path = 'static/face_recognition_model.pkl'
            if os.path.exists(model_path):
                os.remove(model_path)
                print("No users remaining, deleted model file")

        # Notify external API about face removal
        try:
            api_url = f'{API_PROTOCOL}://{API_HOST}:{API_PORT}/members/remove_face_scan'
            api_response = requests.post(
                api_url,
                json={'customer_number': user_id},
                timeout=10
            )
            print(f"API Response ({api_url}): {api_response.status_code}")
        except Exception as e:
            print(f"API notification error (non-blocking): {e}")

        return jsonify({
            'status': 'success',
            'user_id': user_id,
            'message': f'Successfully unregistered {user_id}',
            'remaining_users': len(remaining_users)
        })

    except Exception as e:
        print(f"Error unregistering user: {e}")
        return jsonify({'error': f'Failed to unregister user: {str(e)}'}), 500


#### Proxy endpoint to get member data from external API
@app.route('/members/<customer_number>', methods=['GET'])
def get_member(customer_number):
    try:
        api_url = f'{API_PROTOCOL}://{API_HOST}:{API_PORT}/members/{customer_number}'
        response = requests.get(api_url, timeout=5)
        return jsonify(response.json()), response.status_code
    except requests.exceptions.RequestException as e:
        print(f"Error fetching member data: {e}")
        return jsonify({'error': 'Failed to fetch member data'}), 500


#### Our main function which runs the Flask App
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)