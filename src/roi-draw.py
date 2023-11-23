import cv2
import numpy as np
import datetime

refPt = []

def click_event(event, x, y, flags, param):
    global img, refPt

    if event == cv2.EVENT_LBUTTONDOWN:
        Point = (x, y)
        refPt.append(Point)

        if len(refPt) == 1:
            center_coordinates = refPt[0]
            cv2.line(img, center_coordinates, 5, (0, 255, 0), 5)

        if len(refPt) > 1:
            start_point = refPt[-2]
            end_point = refPt[-1]
            cv2.line(img, start_point, end_point, (0, 255, 0), 9)
        
        print(refPt)

img = cv2.imread('/app/smoke-detect/computer-vision/output/frame/NO20231103-074752-024803/110.png')
original_img = img.copy()

cv2.namedWindow("Image", cv2.WINDOW_NORMAL)

cv2.setMouseCallback("Image", click_event)

while True:
    cv2.imshow("Image", img)

    # Press 'q' to quit
    key = cv2.waitKey(1)
    if key == ord(' '):
        current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f'/app/smoke-detect/computer-vision/output/roi/result_image_{current_time}.jpg'
        cv2.imwrite(filename, img)
        break

cv2.destroyAllWindows()