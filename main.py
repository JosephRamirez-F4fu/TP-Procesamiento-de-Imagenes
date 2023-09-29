import cv2
import pickle
import numpy as np
import cvzone

with open('espacios.pkl', 'rb') as file:
    zones = pickle.load(file)

video = cv2.VideoCapture('video_comedor.mp4')
length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

n_zones = len(zones)
zone_total = np.array([w * h for _, _, w, h in zones])
print(zone_total)


while True:
    ret, frame = video.read()
    if not ret:
        break
    frameBN = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frameTH = cv2.adaptiveThreshold(frameBN, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16)
    frameMedian = cv2.medianBlur(frameTH, 7)
    kernel = np.ones((5,5), np.int8)
    frameDil = cv2.dilate(frameMedian, kernel)
    n_free_zones = 0
    zone_id = 0
    for x, y, w, h in zones:
        zone = frameDil[y:y+h, x:x+w]
        count = cv2.countNonZero(zone)/zone_total[zone_id]
        cv2.putText(frame, "{:.4f}".format(count), (x, y + h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        if count < 0.1:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            n_free_zones += 1
        zone_id += 1
    cvzone.putTextRect(frame, str(n_free_zones) + f"/{n_zones}", (50, 50), 3, 5, offset=20)

    cv2.imshow('FreeZoneDetector', frame)#TH)
    key = cv2.waitKey(10)
    if key == ord('x'):
        break

cv2.destroyAllWindows()
