import cv2
import numpy as np
import time
import os
from cvzone.HandTrackingModule import HandDetector

########################
brushThickness = 15
eraserThickness = 50
########################


folderPath = "header"
myList = os.listdir(folderPath)
print (myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

print(len(overlayList))

header = overlayList[0]
drawColor = (255, 0, 255)

cap = cv2.VideoCapture("https://192.168.1.11:4343/video")
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(detectionCon=0.75, maxHands=1)
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

while True:

    # 1. Import image
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = cv2.resize(img, (1280, 720))

    # 2. Find Hand Landmarks
    hands, img = detector.findHands(img, flipType=False)

    if hands:
        hand = hands[0]

        lmList = hand["lmList"]

        # tip of index and middle fingers
        x1, y1 = lmList[8][:2]
        x2, y2 = lmList[12][:2]


        # 3. Check which fingers are up

        fingers = detector.fingersUp(hand)
        print (fingers)

        # 4. If Selection mode - Two finger are up
        if fingers[1] and fingers[2]:
            print("Selection Mode")
            # Checking for the click
            if y1 < 125:
                if 250<x1<450:
                    header = overlayList[0]
                    drawColor = (255,0,255)
                elif 550<x1<750:
                    header = overlayList[1]
                    drawColor = (255, 0, 0)
                elif 800<x1<950:
                    header = overlayList[2]
                    drawColor = (0, 255, 0)
                elif 1050<x1<1200:
                    header = overlayList[3]
                    drawColor = (0, 0, 0)
            cv2.rectangle(img, (x1, y1 - 15), (x2, y2 + 25), drawColor, cv2.FILLED)
            xp, yp = 0, 0

        # 5. If Drawing Mode - Index finger is up
        if fingers[1] and fingers[2]==False:
            cv2.circle(img, (x1, y1),15,drawColor, cv2.FILLED)
            print("Drawing Mode")
            if (xp==0 and yp == 0):
                xp, yp = x1, y1


            if drawColor == (0,0,0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
            xp, yp = x1, y1
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray,50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    print(img.shape)
    print(imgInv.shape)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)


    # Setting the header image
    img[0:125, 0:1280] = header[0:125, 0:1280]
    #img = cv2.addWeighted(img, 0.5, imgCanvas,0.5,0)
    cv2.imshow("Image", img)
    #cv2.imshow("Canvas", imgCanvas)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break