from sqlite3 import Timestamp
import cv2
import numpy as np
import face_recognition
import pyttsx3 as textToSpeech
import os
from datetime import datetime


path = "student_images"
studentsimg = []
studentNames = []
mylist = os.listdir(path)
# print(mylist)


# loop to read images from saved directory
for cl in mylist:
    curImg = cv2.imread(f'{path}\{cl}')
    studentsimg.append(curImg)
    studentNames.append(os.path.splitext(cl)[0])


# print(studentNames)

def gettingEncoding(images):
    encoding_img_list = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        endcodeimg = face_recognition.face_encodings(img)[0]
        encoding_img_list.append(endcodeimg)
    return encoding_img_list


# instantiating text_to_speech
engine = textToSpeech.init()


def MarkAttendance(name):
    with open("attendance.csv", "r+") as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(",")
            nameList.append(entry[0])
            
        if name not in nameList:
            now = datetime.now()
            timeStr = now.strftime('%H: %M')
            f.writelines(f'\n{name},    {timeStr}')
            statement = str("welcome to class " + name)
            engine.say(statement)
            engine.runAndWait()




encode_list = gettingEncoding(studentsimg)

vid = cv2.VideoCapture(0)

while True:
    ret, frame = vid.read()
    frames = cv2.resize(frame, (0,0), None, 0.25, 0.25)
    frames = cv2.cvtColor(frames, cv2.COLOR_BGR2RGB)

    facesInFrame = face_recognition.face_locations(frames)
    encodeFacesInFrame = face_recognition.face_encodings(frames, facesInFrame)

    for encodeFace, faceLoc in zip(encodeFacesInFrame,facesInFrame):
        matches = face_recognition.compare_faces(encode_list, encodeFace)
        face_dis = face_recognition.face_distance(encode_list, encodeFace)
        print(face_dis)
        matchIndex = np.argmin(face_dis)

        if matches[matchIndex]:
            name =studentNames[matchIndex].upper()
            y1,x1,y2,x2 = faceLoc
            y1,x1,y2,x2 = y1*4 ,x1*4 , y2*4, x2*4

            cv2.rectangle(frame, (x1,y1),(x2,y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (x1,y2-25),(x2,y2), (0,255,0), cv2.FILLED)
            cv2.putText(frame, name, (x1+6,y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            MarkAttendance(name)


    cv2.imshow("video", frame)
    cv2.waitKey(1)