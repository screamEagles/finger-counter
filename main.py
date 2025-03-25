import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector


cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)


while True:
    success, image = cap.read()
    hands, image = detector.findHands(image, draw=False, flipType=True)  # keep the flipType True, or you will face counting problems with the thumb
    if hands:
        hand1 = hands[0]
        landmark_list1 = hand1["lmList"]
        bounding_box1 = hand1["bbox"]
        centre1 = hand1["center"]
        hand_type1 = hand1["type"]

        fingers1 = detector.fingersUp(hand1)
        cvzone.putTextRect(image, f"{fingers1.count(1)}", (50, 50), colorR=(250, 196, 2), offset=10, border=1, colorB=(0, 0, 0))
    
    cv2.imshow("Image", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

