import cv2
import numpy as np

image = cv2.imread('1.KADEME_3P1V3V3403-P1811786-03_frame_0069.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 500)
contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

def calculate_angle(pt1, pt2, pt3):
    a = np.linalg.norm(pt2 - pt3)
    b = np.linalg.norm(pt1 - pt3)
    c = np.linalg.norm(pt1 - pt2)
    
    if a + b <= c or a + c <= b or b + c <= a:
        return 0  

    cosine_value = (b**2 + c**2 - a**2) / (2 * b * c)
    cosine_value = max(-1.0, min(1.0, cosine_value))
    angle = np.arccos(cosine_value)
    return np.degrees(angle)

def calculate_distance(pt1, pt2):
    return np.linalg.norm(pt1 - pt2)

detected_triangles = []

for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.001*cv2.arcLength(cnt, True), True)
    
    if len(approx) > 2:
        for i in range(len(approx)):
            p1 = tuple(approx[i % len(approx)][0])
            p2 = tuple(approx[(i + 1) % len(approx)][0])
            p3 = tuple(approx[(i + 2) % len(approx)][0])

            angle = calculate_angle(np.array(p1), np.array(p2), np.array(p3))
            d1 = calculate_distance(np.array(p1), np.array(p2))
            d2 = calculate_distance(np.array(p2), np.array(p3))
            d3 = calculate_distance(np.array(p3), np.array(p1))
            
            if 20 < angle < 160 and 10 < d1 < 35 and 10 < d2 < 35 and 10 < d3 < 35:
                triangle = [p1, p2, p3]
               
                if triangle not in detected_triangles:
                    
                    cv2.circle(image, p1, 3, (0, 255, 0), -1)
                    cv2.circle(image, p2, 3, (0, 255, 0), -1)
                    cv2.circle(image, p3, 3, (0, 255, 0), -1)
                    cv2.line(image, p1, p2, (255, 0, 0), 1)
                    cv2.line(image, p2, p3, (255, 0, 0), 1)
                    cv2.line(image, p3, p1, (255, 0, 0), 1)
                    
                    
                    detected_triangles.append(triangle)

print(f"Unique triangles detected: {len(detected_triangles)}")
print(f"detected_triangles : {detected_triangles}")
cv2.imshow('Detected Triangles', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
