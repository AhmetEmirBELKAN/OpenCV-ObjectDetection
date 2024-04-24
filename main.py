import cv2
import numpy as np

# frame = cv2.imread('1.KADEME_3P1V3V3403-P1811786-03_frame_0069.jpg')
cap=cv2.VideoCapture("AUNDEDATA/1.KADEME/3P1V3K3395-P1811779-03.wmv")
# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# edges = cv2.Canny(gray, 50, 500)
# contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

detected_triangles = []

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

def FindContours(cap,edges):
        
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for i, contour in enumerate(contours):

            next_contour = hierarchy[0][i][0]
            cv2.drawContours(cap, contours, i, (0, 255, 0), 2)

            if next_contour != -1:
                next_parent_contour = hierarchy[0][next_contour][3]
                parent_contour = hierarchy[0][i][3]
                if parent_contour != -1 and hierarchy[0][parent_contour][3] == -1:
                    area = cv2.contourArea(contour)
                    print(f"area : {area}")  
                    


while True:
    success,frame=cap.read()

    if(success):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 500)
        # FindContours(frame,edges=edges)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        print(f"frame : {frame.shape}")
        for i,cnt in enumerate(contours) :
            # cv2.drawContours(frame, cnt, i, (0, 255, 0), 2)

            approx = cv2.approxPolyDP(cnt, 0.001*cv2.arcLength(cnt, True), True)
            for i in approx:
                print(i)
                # cv2.circle(frame, i[0], 3, (0, 255, 0), -1)
            print(len(approx))
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
                        print(f"angle : {angle}")
                        print(f"d1 : {d1}")
                        print(f"d2 : {d2}")
                        print(f"d3 : {d3}")
                        if triangle not in detected_triangles:
                            
                            cv2.circle(frame, p1, 3, (0, 255, 0), -1)
                            cv2.circle(frame, p2, 3, (0, 255, 0), -1)
                            cv2.circle(frame, p3, 3, (0, 255, 0), -1)
                            cv2.line(frame, p1, p2, (255, 0, 0), 1)
                            cv2.line(frame, p2, p3, (255, 0, 0), 1)
                            cv2.line(frame, p3, p1, (255, 0, 0), 1)
                            
                            
                            detected_triangles.append(triangle)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.imwrite("",frame)
            break
        cv2.imshow("frame",frame)
        cv2.waitKey(200)



detected_triangles = []



print(f"Unique triangles detected: {len(detected_triangles)}")
print(f"detected_triangles : {detected_triangles}")
cv2.imshow('Detected Triangles', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
