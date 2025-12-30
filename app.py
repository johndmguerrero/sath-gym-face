import cv2
import os
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
    global detected_person, detection_start_time, attendance_mode

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
        detected_person = None
        detection_start_time = None
        return jsonify({'status': 'no_face', 'message': 'No face detected'})

    (x, y, w, h) = faces[0]
    face = cv2.resize(frame[y:y + h, x:x + w], (50, 50))

    try:
        identified_person = identify_face(face.reshape(1, -1))[0]
    except Exception as e:
        return jsonify({'error': f'Recognition error: {str(e)}'}), 500

    if detected_person != identified_person:
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
            api_url = f'http://{API_HOST}:{API_PORT}/attendances'
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
            api_url = f'http://{API_HOST}:{API_PORT}/members/update_face_scan'
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


#### Proxy endpoint to get member data from external API
@app.route('/members/<customer_number>', methods=['GET'])
def get_member(customer_number):
    try:
        api_url = f'http://{API_HOST}:{API_PORT}/members/{customer_number}'
        response = requests.get(api_url, timeout=5)

        if response.status_code == 200:
            return jsonify(response.json()), 200
        else:
            return jsonify({'error': 'Member not found'}), response.status_code
    except requests.exceptions.RequestException as e:
        print(f"Error fetching member data: {e}")
        return jsonify({'error': 'Failed to fetch member data'}), 500


#### Our main function which runs the Flask App
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)