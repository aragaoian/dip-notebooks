import cv2
import face_recognition
import os
import time
from datetime import datetime
import sqlite3 as sql

# Creation of the db to store the name ocurrences
# connection = sql.connect('test.db')
# cursor = connection.cursor()
# cursor.execute('''CREATE TABLE IF NOT EXISTS processamento_frames (
#                id_frame PRIMARY KEY AUTOINCREMENT, 
#                nome TEXT NOT NULL, 
#                data_hora TEXT NOT NULL)''')
# connection.commit()

# Retrieve data from the db and write in a .txt
# def retrieve_info_fromDB():
#     cursor.execute('''SELECT nome, COUNT(nome) AS freq FROM processamento_frames
#                    GROUP BY nome
#                    ORDER BY freq DESC
#                    ''')
#     res = cursor.fetchall()
#     file = open("res.txt", "a+", encoding="utf-8")
#     for row in res:
#         file.write(f"Nome: {row[0]}, Frequência: {row[1]}\n")
    

path_dataset = os.path.dirname(cv2.__file__) + '/data/haarcascade_frontalface_alt2.xml'
# print(os.path.exists(path_dataset))
cascade_classifier = cv2.CascadeClassifier(path_dataset)

known_encodings = []
known_names = []
face_directory = '/home/ian/Univali/Machine Learning/ComputerVision_Projects/FaceRecognition/faces'

for filename in os.listdir(face_directory):
    img_path = os.path.join(face_directory, filename)
    # Ensure you're only processing image files (e.g., .jpg, .png)
    if filename.endswith('.jpg') or filename.endswith('.png'):
        name = filename.split('.')[0]
        known_names.append(name)
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(img_rgb, model='hog') # locate face on the img
        encodings = face_recognition.face_encodings(img_rgb, boxes, model="small") # facial embedding for the faces
        for encoding in encodings:
            known_encodings.append(encoding)

data = {'encodings': known_encodings, 'names': known_names}

cap = cv2.VideoCapture('/dev/video0')
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

process_frame_check = True

if not cap.isOpened():
    print('Error')
else:
    while True:
        check, frame = cap.read()
        if not check: break
        start = time.time()

        if process_frame_check:

            matched_names = []
            resized_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)

            gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
            faces = cascade_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5) # ,flags=cv2.CASCADE_SCALE_IMAGE) # returns a list of rectangles corresponding to the detected faces
            
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb_frame, model='small') # list of extracted face features

            for encoding in encodings:
                matches = face_recognition.compare_faces(data['encodings'], encoding)
                mostFreq_name = 'Unkown'
                if True in matches:
                    ids = [id for id, match in enumerate(matches) if match]
                    matchedFaces_counts = {}
                    for id in ids:
                        curr_name = data['names'][id]
                        matchedFaces_counts[curr_name] = matchedFaces_counts.get(name, 0) + 1
                    mostFreq_name = max(matchedFaces_counts, key=matchedFaces_counts.get)
                matched_names.append(mostFreq_name) # append the most matched name in the entire frame (could be more than 1)

        if len(faces) == len(matched_names):
            for ((x, y, w, h), name) in zip (faces, matched_names): # x-top_left | y-top_left
                x*=4
                y*=4
                w*=4
                h*=4
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,0.75, (255, 255, 255), 2)

        process_frame_check = not process_frame_check

        end = time.time()
        fps = str(int(1/ (end-start)))
        cv2.putText(frame, fps, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2) 

        cv2.imshow('face-recognition', frame)
        key = cv2.waitKey(1)
        if key == 27:
            break

cap.release()
cv2.destroyAllWindows()

# retrieve_info_fromDB()
