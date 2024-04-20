import cv2
import numpy as np

# Resmi yükle
image = cv2.imread('1.KADEME-3P1V3K3395-P1811778-03.wmv-_frame1.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 200)
contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

def angle_between(p1, p2, p0):  # p0: ortak nokta
    v1 = p1 - p0
    v2 = p2 - p0
    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.001*cv2.arcLength(cnt, True), True)
    print(len(approx))
    for i in approx:
        print(i)
        cv2.circle(image,i[0],5,(255,0,0),3)
    if len(approx) == 3:
        pt1, pt2, pt3 = approx.reshape(3, 2)
        angle1 = angle_between(pt2, pt3, pt1)
        angle2 = angle_between(pt1, pt3, pt2)
        angle3 = angle_between(pt1, pt2, pt3)

        # Açıların toplamı yaklaşık 180 derece ve her biri 0'dan büyük ve 180'den küçükse bu bir üçgendir
        if angle1 > 0 and angle2 > 0 and angle3 > 0 and (angle1 + angle2 + angle3 - 180) < 10:
            cv2.polylines(image, [approx], True, (0, 255, 0), 3)

cv2.imshow('Triangle Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()