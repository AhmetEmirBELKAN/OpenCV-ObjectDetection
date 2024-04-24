import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow
from design import Ui_MainWindow 
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import os
from pathlib import Path
from PyQt5.QtCore import QThread, pyqtSignal
import time


import PyQt5
import cv2
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.fspath(
    Path(PyQt5.__file__).resolve().parent / "Qt5" / "plugins"
)
class ImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
    
    def initUI(self):
        self.imageLabel = QLabel(self)
        self.imageLabel.setFixedSize(1280, 720)
        
        self.btnOpenImage = QPushButton('Open Image', self)
        
        layout = QVBoxLayout()
        layout.addWidget(self.imageLabel)
        layout.addWidget(self.btnOpenImage)
        
        centralWidget = QWidget()
        centralWidget.setLayout(layout)
        self.setCentralWidget(centralWidget)
        
        self.setWindowTitle('Image Viewer')


class MainThread(QThread):
    data_received = pyqtSignal(str)
    
    def __init__(self, main):
        super().__init__()
        self.Main = main
        self.running = True

   
    def run(self):
        while self.running:
            try:
                
                self.Main()
                time.sleep(0.1)  

            except Exception as e:
                print("Error reading data:", e)

    def stop(self):
        self.running = False

class MainWindow(QMainWindow,Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.detected_triangles=[]
        self.CurrentImage=None
        self.data_reader_thread = None

        # Button Clicked Event
        self.ui.btn_open.clicked.connect(self.Openimage)
        self.ui.btn_opendir.clicked.connect(self.Opendir)
        self.ui.btn_change_save_dir.clicked.connect(self.CurrentSaveDirChange)
        self.ui.btn_next_image.clicked.connect(self.NextImage)
        self.ui.btn_prev_image.clicked.connect(self.PrevImage)
        self.ui.btn_Save.clicked.connect(self.SaveDataFormat)
        self.CutDetection_Is_Start=False

        # Slider Tracker Event

        # High-H Low-H
        self.ui.highH_trackbar.valueChanged.connect(self.HighH_trackbar)
        self.ui.lowH_trackbar.valueChanged.connect(self.LowH_trackbar)
        
        # High-S Low-S
        self.ui.highS_trackbar.valueChanged.connect(self.HighS_trackbar)
        self.ui.lowS_trackbar.valueChanged.connect(self.LowS_trackbar)

        # High-V Low-V
        self.ui.highV_trackbar.valueChanged.connect(self.HighV_trackbar)
        self.ui.lowV_trackbar.valueChanged.connect(self.LowV_trackbar)

        # Thresh_min Thresh_max
        self.ui.thresh_max_trackbar.valueChanged.connect(self.thresh_max_trackbar)
        self.ui.thresh_min_trackbar.valueChanged.connect(self.thresh_min_trackbar)

        # Kernel_size
        self.ui.kernel_size_trackbar.valueChanged.connect(self.kernel_size_trackbar)

        # Erode
        self.ui.erode_trackbar.valueChanged.connect(self.erode_trackbar)

        # Dilade 
        self.ui.dilate_trackbar.valueChanged.connect(self.dilate_trackbar)

        # MORPH_OPEN
        self.ui.MORPH_OPEN_trackbar.valueChanged.connect(self.MORPH_OPEN_trackbar)

        # MORPH_CLOSE
        self.ui.MORPH_CLOSE_trackbar.valueChanged.connect(self.MORPH_CLOSE_trackbar)

        # Morpholojic Value
        self.ilowH = 0
        self.ihighH = 255
        self.thresh_max=255
        self.thresh_min=0
        self.ilowS = 0
        self.ihighS = 255
        self.ilowV = 0
        self.ihighV = 255
        self.erode=0
        self.dilate=0
        self.MORPH_OPEN=0
        self.MORPH_CLOSE=0
        self.kernel_size=5
        self.kernel = np.ones((self.kernel_size,self.kernel_size), np.uint8)

    def Openimage(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        print(f"Directory path: {dir_path}")
        origin_images_label_label_h=self.ui.origin_images_label.height()
        origin_images_label_label_w=self.ui.origin_images_label.width()
        Object_Detection_Canvas_h=self.ui.Object_Detection_Canvas.height()
        Object_Detection_Canvas_w=self.ui.Object_Detection_Canvas.width()
        fname, _ = QFileDialog.getOpenFileName(self, 'Open file', dir_path, "Image files (*.jpg *.gif *.png)")
        print(f"File selected: {fname}")
        self.FrameLoadPixmap(path=fname,size=(origin_images_label_label_w,origin_images_label_label_h))
        self.FrameLoadPixmap(path=fname,size=(Object_Detection_Canvas_w,Object_Detection_Canvas_h),IsAlive=True)
        self.data_reader_thread = MainThread(self.Main)
        self.data_reader_thread.start()

    def calculate_angle(self,pt1, pt2, pt3):
        a = np.linalg.norm(pt2 - pt3)
        b = np.linalg.norm(pt1 - pt3)
        c = np.linalg.norm(pt1 - pt2)
        
        if a + b <= c or a + c <= b or b + c <= a:
            return 0  

        cosine_value = (b**2 + c**2 - a**2) / (2 * b * c)
        cosine_value = max(-1.0, min(1.0, cosine_value))
        angle = np.arccos(cosine_value)
        return np.degrees(angle)

    def calculate_distance(self,pt1, pt2):
        return np.linalg.norm(pt1 - pt2)
    
    def CutDetection(self,frame):
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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

                    angle = self.calculate_angle(np.array(p1), np.array(p2), np.array(p3))
                    d1 = self.calculate_distance(np.array(p1), np.array(p2))
                    d2 = self.calculate_distance(np.array(p2), np.array(p3))
                    d3 = self.calculate_distance(np.array(p3), np.array(p1))
                    
                    if 20 < angle < 160 and 10 < d1 < 35 and 10 < d2 < 35 and 10 < d3 < 35:
                        triangle = [p1, p2, p3]
                        print(f"angle : {angle}")
                        print(f"d1 : {d1}")
                        print(f"d2 : {d2}")
                        print(f"d3 : {d3}")
                        if triangle not in self.detected_triangles:
                            
                            cv2.circle(frame, p1, 3, (0, 255, 0), -1)
                            cv2.circle(frame, p2, 3, (0, 255, 0), -1)
                            cv2.circle(frame, p3, 3, (0, 255, 0), -1)
                            cv2.line(frame, p1, p2, (255, 0, 0), 1)
                            cv2.line(frame, p2, p3, (255, 0, 0), 1)
                            cv2.line(frame, p3, p1, (255, 0, 0), 1)
                            
                            self.FrameLoadPixmap(IsAlive=True)
                            self.detected_triangles.append(triangle)

    def Main(self):
        self.update_image_processing()

    def update_image_processing(self):
        if self.CurrentImage is not None:
            gray =cv2.cvtColor(self.CurrentImage, cv2.COLOR_BGR2RGB)

            _, thresholded = cv2.threshold(gray, self.thresh_min, self.thresh_max, cv2.THRESH_BINARY)
            modified_image = cv2.erode(thresholded, self.kernel, iterations=self.erode)
            modified_image = cv2.dilate(modified_image, self.kernel, iterations=self.dilate)

            if self.MORPH_OPEN > 0:
                modified_image = cv2.morphologyEx(modified_image, cv2.MORPH_OPEN, self.kernel)
            if self.MORPH_CLOSE > 0:
                modified_image = cv2.morphologyEx(modified_image, cv2.MORPH_CLOSE, self.kernel)
            modified_image=self.FindContour(modified_image)

            self.FrameLoadPixmap(IsAlive=True, image=modified_image)        
            
    def FindContour(self, frame):
        # blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        
        edges = cv2.Canny(frame, 50, 150)
        # print(f"edges : {edges}")
        # Find contours from the edged image
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"Number of contours found: {len(contours)}")
        cv2.drawContours(frame, contours, -1, (0, 255, 0), 1)


        # cv2.drawContours function draws detected contours on the original colored image
        # cv2.drawContours(frame, contours, -1, (0, 255, 255), 2)
        
        return frame
    
    def Opendir(self):
        # file = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        pass

    def CurrentSaveDirChange(self):
        pass
    
    def NextImage(self):
        pass
    
    def PrevImage(self):
        pass

    def SaveDataFormat(self):
        pass
    
    def HighH_trackbar(self,value):
        self.ihighH=value

    def LowH_trackbar(self,value):
        self.ilowH=value
    
    def HighS_trackbar(self,value):
        self.ihighS=value

    def LowS_trackbar(self,value):
        self.ilowS=value

    def HighV_trackbar(self,value):
        self.ihighV=value

    def LowV_trackbar(self,value):
        self.ilowV=value
    
    def thresh_max_trackbar(self,value):
        self.thresh_max=value
    
    def thresh_min_trackbar(self,value):
        self.thresh_min=value

    def kernel_size_trackbar(self,value):
        self.kernel_size=value
    
    def erode_trackbar(self,value):
        self.erode=value
    
    def dilate_trackbar(self,value):
        self.dilate=value

    def MORPH_OPEN_trackbar(self,value):
        self.MORPH_OPEN=value
        print(f"MORPH_OPEN : {value}")

    def MORPH_CLOSE_trackbar(self,value):
        self.MORPH_CLOSE=value
        print(f"MORPH_CLOSE : {value}")

    def FrameLoadPixmap(self, path=None, size=(1280,720), IsAlive=False, image=None):
        if image is not None:
            img_rgb = image
            if len(img_rgb.shape) == 2:  # It's a grayscale image
                img_rgb =cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_rgb = cv2.resize(img_rgb, size)
        elif path:
            frame = cv2.imread(path)
            if frame is not None:
                print("Frame loaded successfully.")
                if len(frame.shape) == 2:  # Check if the loaded image is grayscale
                    frame =cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.CurrentImage = img_rgb
                print(f"img_rgb_type : {type(img_rgb)}")
                img_rgb = cv2.resize(img_rgb, size)
            else:
                print("Failed to load the image.")
                return
        else:
            print("No file selected or image data provided.")
            return

        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        q_img = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        if IsAlive:
            self.ui.Object_Detection_Canvas.setPixmap(pixmap)
        else:
            self.ui.origin_images_label.setPixmap(pixmap)
def LoadUi():
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()
    image_viewer = ImageViewer()

    main_window.show()
    image_viewer.show()

    sys.exit(app.exec_())  

if __name__ == "__main__":
    LoadUi()