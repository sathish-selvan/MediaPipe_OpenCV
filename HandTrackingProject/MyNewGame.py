import cv2
import time
from HandTrackingProject import HandTrackingModule as htm

cap = cv2.VideoCapture(0)
pTime = 0
cTime = 0
detector = htm.HandDetector()
while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList  = detector.find_position(img)

    if len(lmList) != 0:
        print(lmList[4])

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 8, 255), 3)
    img = img[:, ::-1]
    cv2.imshow("Image", img)
    cv2.waitKey(1)

