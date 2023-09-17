import json
import face_recognition
import cv2
import os
import datetime
import numpy as np

# Finds encodings for the face in the images to be used for facial recognition
def getEncodings(images):
    encodings = []
    for img in images:
        encode = face_recognition.face_encodings(img)[0] # Store the encoding of the main face in the image
        encodings.append(encode)
    return encodings
 

# If a person's attendance is not in file, add their name to the file with the date of their attendance,
# Otherwise just update their attendance and time for the existing entry
def updatePersonAttendanceInfo(name):
    with open('src/Attendance.json', 'r+') as file:
        attendees = json.load(file)
        names = []

        # Gather all names in attendance
        for person in attendees:
            names.append(person["name"])

            if(person["name"] == name):
                if(person["attendance"] == "false"):
                    currentTimeDate = datetime.datetime.now()
                    currentTime = currentTimeDate.strftime('%H:%M:%S')
                    person["attendance"] = "true"
                    person["attendanceTime"] = currentTime

        if name not in names:
            currentTimeDate = datetime.datetime.now()
            currentTime = currentTimeDate.strftime('%H:%M:%S')
            newEntry = {
                "name": name,
                "attendance": "true",
                "attendanceTime": currentTime
            }
            attendees.append(newEntry)

    with open("src/Attendance.json", 'w') as file:
        json.dump(attendees, file, indent=4, separators=(',',': '))


# Clear attendance file
with open("src/Attendance.json", 'r') as file:
    attendees = json.load(file)

for person in attendees:
    person["attendance"] = 'false'
    person["attendanceTime"] = "00:00:00"

with open("src/Attendance.json", 'w') as file:
    json.dump(attendees, file, indent=4)


# Gather all people and images and face encodings for saved images of members
folder = 'Images'
images = []
names = []
imgFiles = os.listdir(folder) # Collect all image file names
for file in imgFiles:
    currentImg = cv2.imread(f'{folder}/{file}') # read the image file using its path name
    images.append(currentImg)
    names.append(os.path.splitext(file)[0]) # get the image's file name (person name) without extension
 
faceEncodings = getEncodings(images)
 

# Create a video capture object
capture = cv2.VideoCapture(0)
 

while True:
    success, img = capture.read() # Captures a frame from the video and stores it in img

    imgResized = cv2.resize(img, (0,0), None, 0.25, 0.25)
 
    # Store the locations and face encodings for the faces in the video capture
    faceLocations = face_recognition.face_locations(imgResized)
    facesDisplayedEncodings = face_recognition.face_encodings(imgResized, faceLocations)
 
    # Loop through the detected faces and their encodings
    for encode, location in zip(facesDisplayedEncodings,faceLocations):
        # Compare the encoding of the face in the capture to the stored face encodings in Images
        faceComparisons = face_recognition.compare_faces(faceEncodings, encode)

        # Evaluates how similar faces are to the stored face images
        faceSimilarities = face_recognition.face_distance(faceEncodings, encode)

        # Get index of the face with the closest similarity
        minIndex = np.argmin(faceSimilarities)
 
        # If a match found, post label on person's face and update their attendance info on file
        if faceComparisons[minIndex]:
            name = names[minIndex].upper()
            y1, x2, y2, x1 = location
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0), 2)
            cv2.rectangle(img, (x1,y2-50), (x2,y2), (255,0,0), cv2.FILLED)
            cv2.putText(img, name, (x1+35,y2-5), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)
            updatePersonAttendanceInfo(name)
 
    cv2.imshow('Face Recognition', img)
    cv2.waitKey(1)

