import cv2
import numpy as np
# Resmi yükle
image = cv2.imread('1.KADEME-3P1V3K3395-P1811778-03.wmv-_frame1.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 200)
contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Açı hesaplama fonksiyonu
def calculate_angle(pt1, pt2, pt3):
    a = np.linalg.norm(pt2 - pt3)
    b = np.linalg.norm(pt1 - pt3)
    c = np.linalg.norm(pt1 - pt2)
    angle = np.arccos((b**2 + c**2 - a**2) / (2 * b * c))
    return np.degrees(angle)

# Konturlar için döngü
for cnt in contours:
    # Her kontur için approx hesapla
    approx = cv2.approxPolyDP(cnt, 0.001*cv2.arcLength(cnt, True), True)
    y=0
    # approx noktalarında döngü
    print(f"len(approx) : {len(approx)}")
    for i in range(len(approx)):
        # Üç ardışık noktayı al
        p1 = approx[i % len(approx)][0]
        p2 = approx[(i + 1) % len(approx)][0]
        p3 = approx[(i + 2) % len(approx)][0]

        # Açıyı hesapla
        angle = calculate_angle(p1, p2, p3)
        print(angle)
        # Açı eğer bir üçgen köşesi ise
        
        if 20 < angle < 160:  # Üçgen köşe açıları için uygun aralık
            print("üçgennnnnnnnnnn")
            y+=1
            # Noktaları çiz
            cv2.circle(image, tuple(p1), 3, (0, 255, 0), -1)
            cv2.circle(image, tuple(p2), 3, (0, 255, 0), -1)
            cv2.circle(image, tuple(p3), 3, (0, 255, 0), -1)
            # Üçgen köşe noktalarını birleştir
            cv2.line(image, tuple(p1), tuple(p2), (255, 0, 0), 1)
            cv2.line(image, tuple(p2), tuple(p3), (255, 0, 0), 1)
            cv2.line(image, tuple(p3), tuple(p1), (255, 0, 0), 1)
    print(y)
# Görselleştir
cv2.imshow('Detected Triangles', image)
cv2.waitKey(0)
cv2.destroyAllWindows()