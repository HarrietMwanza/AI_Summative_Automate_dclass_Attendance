import csv
import cv2
import face_recognition
import numpy as np
import datetime
import os

# Load images and encode faces
images_dir = 'images'
recognized_faces = []
name_mapping = {
    'Percy.jpeg':'Percy',
    'Hariet.jpeg':'Hariet'
}

for filename in os.listdir(images_dir):
    image = face_recognition.load_image_file(os.path.join(images_dir, filename))
    face_encoding = face_recognition.face_encodings(image)[0]
    recognized_faces.append(face_encoding)
    name_mapping[os.path.splitext(filename)[0]] = os.path.splitext(filename)[0]

# Initialize variables
attendance = {}
        # If a match was found in the known face(s), add the name to the attendance list
if True in name_mapping:
            match_index = name_mapping.index(True)
            name = "Person {}".format(match_index + 1)
            
        # Otherwise, add the new face encoding to the known face(s)
else:
            recognized_faces.append(face_encoding)
            name = "Person {}".format(len(recognized_faces))
        
        # Add the name to the attendance list with the current time
attendance[name] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S present")


# Start video capture
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Find all the faces in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, model='large')

    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(recognized_faces, face_encoding)

        # If a match was found in the known face(s), add the name to the attendance list
        if True in matches:
            match_index = matches.index(True)
            name = name_mapping.get(os.path.splitext(os.listdir(images_dir)[match_index])[0], "Person {}".format(len(recognized_faces)))
        # Otherwise, add the new face encoding to the known face(s)
        else:
            recognized_faces.append(face_encoding)
            name = "Person {}".format(len(recognized_faces))
        
        # Add the name to the attendance list with the current time
        attendance[name] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Display the resulting image
    for (top, right, bottom, left), name in zip(face_locations, attendance.keys()):
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with the name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Attendance System', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the window
video_capture.release()
cv2.destroyAllWindows()

# Write attendance list to a CSV file
with open('attendance.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Name', 'Time'])
    for name, time in attendance.items():
        writer.writerow([name, time])
