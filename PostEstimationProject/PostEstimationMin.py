import cv2
import mediapipe as mp
import time

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
pTime = 0
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)


    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks,mpPose.POSE_CONNECTIONS)
        for id, lm in  enumerate(results.pose_landmarks.landmark):
            h,w,c = img.shape
            cx,cy = int(lm.x*w), int(lm.y*h)

            cv2.circle(img, (cx,cy), 5 , (255,8,255),cv2.FILLED)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime =  cTime

    cv2.putText(img, str(int(fps)),(10,70),cv2.FONT_HERSHEY_SIMPLEX,3,(255, 0, 0), 3)
    img = img[:, ::-1]
    cv2.imshow("image", img)
    cv2.waitKey(1)