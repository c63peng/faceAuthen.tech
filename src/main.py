import json
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# Create empty attendance file
# with open("Attendance.csv", 'r+') as file:
#     people = file.readlines()

#     # Set all people's attendance to false
#     for person in people:
#         print(person)
#         person.replace("TRUE", "jdfkghkdrgjhlerjg")
#         print(person)
#         # personInfo = person.split(',')
#         # personInfo[1] = "FALSE"
#     file.writelines(people)
#     # file.truncate(0)
with open("Attendance.json", 'r') as file:
    attendees = json.load(file)
    attendees = attendees["attendees"]
    
for person in attendees:
    print(person)
    print("hi")
    person.attendance = 'false'

with open("Attendance.json", 'w') as file:
    json.dump(attendees, file, indent=4)


# Gather all people and images
folder = 'Images'
images = []
names = []
myList = os.listdir(folder) # Collect all image file names
for person in myList:
    curImg = cv2.imread(f'{folder}/{person}') # read the image file using its path name
    images.append(curImg)
    names.append(os.path.splitext(person)[0]) # get the image's file name (person name) without extension
 

# Finds encodings for the face in the images to be used for facial recognition
def findEncodings(images):
    encodeList = []
    for img in images:
        encode = face_recognition.face_encodings(img)[0] # Store the encoding of the main face in the image
        encodeList.append(encode)
    return encodeList
 

# If a person's attendance is not in file, add their name to the file with the date of their attendance
def markAttendance(name):
    with open('Attendance.csv', 'r+') as file:
        attendees = file.readlines()
        names = []
        # Gather all names in attendance
        for person in attendees:
            personAttendance = person.split(',')
            names.append(personAttendance[0])
        if name not in names:
            currentTimeDate = datetime.now()
            currentTime = currentTimeDate.strftime('%H:%M:%S')
            file.writelines(f'\n{name}, {currentTime}')

 
faceEncodings = findEncodings(images)
 
# Create a video capture object
capture = cv2.VideoCapture(0)
 
while True:
    success, img = capture.read()

    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
 
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)
 
    for encodeFace, faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(faceEncodings,encodeFace)
        faceDis = face_recognition.face_distance(faceEncodings,encodeFace)
        matchIndex = np.argmin(faceDis)
 
        if matches[matchIndex]:
            name = names[matchIndex].upper()
            print(name)
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1), (x2,y2), (0,255,0), 2)
            cv2.rectangle(img,(x1,y2-35), (x2,y2), (0,255,0), cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)
            markAttendance(name)
 
    cv2.imshow('Webcam',img)
    cv2.waitKey(1)