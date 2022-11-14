# importing libraries

import cv2
import numpy as np
import face_recognition as face_rec


# Image resize function
def resize(img, size):
    width = int(img.shape[1]*size)
    height = int(img.shape[0]*size)
    dimension = (width,height)
    return cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)


#img declaration 
linus = face_rec.load_image_file('images/linus.jpg')
linus = cv2.cvtColor(linus, cv2.COLOR_BGR2RGB)

linus_test = face_rec.load_image_file('images/kenneth.webp')
linus_test = cv2.cvtColor(linus_test, cv2.COLOR_BGR2RGB)


#finding face location

faceLocat_linus = face_rec.face_locations(linus)[0]
encodeface_linus = face_rec.face_encodings(linus)[0]
cv2.rectangle(linus, (faceLocat_linus[3],faceLocat_linus[0],faceLocat_linus[1],faceLocat_linus[2]),(255,0,255), 2)

faceLocat_linus_Test = face_rec.face_locations(linus_test)[0]
encodeface_linus_Test = face_rec.face_encodings(linus_test)[0]
cv2.rectangle(linus_test, (faceLocat_linus_Test[3],faceLocat_linus_Test[0],faceLocat_linus_Test[1],faceLocat_linus_Test[2]),(255,0,255), 2)


results = face_rec.compare_faces([encodeface_linus], encodeface_linus_Test)
print(results)
cv2.putText(linus_test,f'{results}',(50,50), cv2.FONT_HERSHEY_COMPLEX, 1,(0,0, 255), 2)

cv2.imshow("Main_image", linus)
cv2.imshow("test_image",linus_test)
cv2.waitKey(0)
cv2.destroyAllWindows() 