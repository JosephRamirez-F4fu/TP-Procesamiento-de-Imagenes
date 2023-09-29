import cv2
import pickle


def draw_zones(img, zones):
    for x, y, w, h in zones:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)


def select_zones(img_path: str, lim: int):
    video = cv2.VideoCapture(img_path)
    ret, img = video.read()
    if not ret:
        exit()
    zones = list()
    for x in range(lim):
        zone = cv2.selectROI('zone selector', img, False)
        cv2.destroyWindow('zone selector')
        print(zone)
        if zone[2] == 0 and zone[3] == 0:
            print("Zone selection process has been stopped.")
            break
        else:
            print("A zone has been selected.")
        zones.append(zone)
        draw_zones(img, zones)
    return zones


selected_zones = select_zones('video_comedor.mp4', 40)
# create file
with open('espacios.pkl', 'wb') as file:
    pickle.dump(selected_zones, file)
