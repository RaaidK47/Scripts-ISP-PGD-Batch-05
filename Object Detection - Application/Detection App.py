import sys, os
import re
import time
import numpy as np
import cv2
from PyQt5.QtWidgets import (QApplication, QHBoxLayout, QVBoxLayout, QWidget, QTextEdit, QFrame, QLabel,QComboBox)
from PyQt5 import QtGui
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import *
from PyQt5.QtWidgets import QPushButton
import time

global yoloVersion
yoloVersion = "YOLOv9"

global version_num
version_num = 9

global device_number
device_number = 0

class CameraThread(QThread):
    # Signal to update the GUI with new frame
    frame_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        global device_number
        super().__init__()
        self.is_running = False
        self.cap = cv2.VideoCapture(device_number)  # Initialize the camera

    def run(self):
        while self.is_running:
            ret, frame = self.cap.read()
            if ret:
                # Emit signal with the captured frame
                self.frame_signal.emit(frame)

    def start_capture(self):
        self.is_running = True
        self.start()

    def stop_capture(self):
        self.is_running = False
        self.wait()  # Wait for the thread to finish before closing

    def release(self):
        self.cap.release()

class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        QWidget.__init__(self)
        self.title = "Object Detection App - ISP - PGD-B05"

        self.InitWindow()

        self.is_video_running = False  # Flag to control video feed

    def updateLogScreen(self, text, color):
        self.sysLogs.setTextColor(color)
        self.sysLogs.insertPlainText(text+"\n")
        self.sysLogs.verticalScrollBar().setSliderDown(1)
        self.sysLogs.moveCursor(QtGui.QTextCursor.End)
        QCoreApplication.processEvents() 

    def InitWindow(self):

        # Disable maximize and resizing
        self.setWindowFlags(Qt.Window | Qt.WindowMinimizeButtonHint | Qt.WindowCloseButtonHint)

        dbFont = QtGui.QFontDatabase()
        boldFont = QtGui.QFont()

        self.setWindowIcon(QtGui.QIcon("./Data/Icon.png")) #Can use .png files also
        self.setWindowTitle(self.title)
		
        self.frameLeft = QFrame()
        self.leftLayout = QVBoxLayout()
        self.frameLeft.setLayout(self.leftLayout)
        self.frameLeft.setFixedWidth(425)

        self.frameMid = QFrame() 
        self.midLayout = QVBoxLayout()
        self.frameMid.setLayout(self.midLayout)
        self.frameMid.setFixedWidth(1025)
		
        self.frameRight = QFrame() 
        self.rightLayout = QVBoxLayout()
        self.frameRight.setLayout(self.rightLayout)

        self.main_frame = QFrame(self)
        self.main_layout = QHBoxLayout()
        self.main_frame.setLayout(self.main_layout)

        # <<--Designing Left Layout-->>  

        # Showing App Logo
        self.logo_label = QLabel()
        self.logo = QPixmap('./Data/App Logo.png')
        self.logo_label.setPixmap(self.logo)
        self.logo_label.resize(self.logo.width(),
                          self.logo.height())
        self.logo_label.setAlignment(Qt.AlignCenter)
        self.logo_label.setContentsMargins(0,10,0,10)
        # self.logo_label.setFixedSize(200, 200)
        self.leftLayout.addWidget(self.logo_label)  
        self.leftLayout.addSpacing(30)
         
        label0 = QLabel()
        label0.setText("Object Detection Application")
        label0.setAlignment(Qt.AlignCenter)
        label0.setStyleSheet("font-size: 27px; font-weight: bold;")
        label0.setContentsMargins(0,0,0,10)
        self.leftLayout.addWidget(label0)
        self.leftLayout.addSpacing(40)
	
        # <-- YOLO Version Selector-->
        # Create a label
        YOLOlabel = QLabel('YOLO Version:', self)
        YOLOlabel.setStyleSheet("font-size: 20px; font-weight: bold;")
        # Create a dropdown
        YOLOdropdown = QComboBox(self)
        YOLOdropdown.addItems(['YOLOv8', 'YOLOv9', 'YOLOv10', 'YOLOv11'])
        YOLOdropdown.setCurrentIndex(1) # Setting YOLOv9 by Default
        YOLOdropdown.setStyleSheet("font-size: 18px;")
        # Create a horizontal layout
        YOLOlayout = QHBoxLayout()
        YOLOlayout.addWidget(YOLOlabel)
        YOLOlayout.addWidget(YOLOdropdown)
        self.leftLayout.addLayout(YOLOlayout)
        self.leftLayout.addSpacing(30)

        def on_version_changed():
            global yoloVersion
            global version_num

            yoloVersion = YOLOdropdown.currentText()

            # Update the model dropdown based on the selected version
            Modeldropdown.clear()
            if yoloVersion == 'YOLOv8':
                Modeldropdown.addItems(['YOLOv8n', 'YOLOv8s', 'YOLOv8m', 'YOLOv8l', 'YOLOv8x'])
                Modeldropdown.setCurrentIndex(1)
            elif yoloVersion == 'YOLOv9':
                Modeldropdown.addItems(['YOLOv9t', 'YOLOv9s', 'YOLOv9m', 'YOLOv9c', 'YOLOv9e'])
                Modeldropdown.setCurrentIndex(1)
            elif yoloVersion == 'YOLOv10':
                Modeldropdown.addItems(['YOLOv10n', 'YOLOv10s', 'YOLOv10m', 'YOLOv10l', 'YOLOv10x'])
                Modeldropdown.setCurrentIndex(1)
            elif yoloVersion == 'YOLOv11':
                Modeldropdown.addItems(['YOLOv11n', 'YOLOv11s', 'YOLOv11m', 'YOLOv11l', 'YOLOv11x'])
                Modeldropdown.setCurrentIndex(1)

        # Connect the currentTextChanged signal to the slot
        YOLOdropdown.currentIndexChanged.connect(on_version_changed)

        # <-- YOLO Model Selector-->
        # Create a label
        Modellabel = QLabel('YOLO Model:', self)
        Modellabel.setStyleSheet("font-size: 20px; font-weight: bold;")
        # Create a dropdown
        Modeldropdown = QComboBox(self)
        Modeldropdown.addItems(['YOLOv9t', 'YOLOv9s', 'YOLOv9m', 'YOLOv9c', 'YOLOv9e'])
        Modeldropdown.setStyleSheet("font-size: 18px;")
        # Create a horizontal layout
        Modellayout = QHBoxLayout()
        Modellayout.addWidget(Modellabel)
        Modellayout.addWidget(Modeldropdown)
        self.leftLayout.addLayout(Modellayout)
        self.leftLayout.addSpacing(30)

        # <-- Device Selector-->
        # Create a label
        Devicelabel = QLabel('Device:', self)
        Devicelabel.setStyleSheet("font-size: 20px; font-weight: bold;")
        # Create a dropdown
        Devicedropdown = QComboBox(self)
        Devicedropdown.addItems(['CPU', 'GPU'])
        Devicedropdown.setCurrentIndex(1)
        Devicedropdown.setStyleSheet("font-size: 18px;")
        # Create a horizontal layout
        Devicelayout = QHBoxLayout()
        Devicelayout.addWidget(Devicelabel)
        Devicelayout.addWidget(Devicedropdown)
        self.leftLayout.addLayout(Devicelayout)
        self.leftLayout.addSpacing(30)


        #<-- System Logs-->
        logLabel = QLabel()
        logLabel.setText("System Logs")
        logLabel.setAlignment(Qt.AlignCenter)
        # logLabel.setFixedSize(330,35)
        logLabel.setFont(QtGui.QFont("ROLNER", 11))
        logLabel.setContentsMargins(0,5,0,0)
        self.leftLayout.addWidget(logLabel)
        self.leftLayout.addSpacing(10)

        self.sysLogs = QTextEdit()
        self.sysLogs.setReadOnly(True)
        self.sysLogs.setFixedSize(400,360)
        self.sysLogs.setFont(QtGui.QFont("arial-unicode-ms", 9))
        self.leftLayout.addWidget(self.sysLogs)

        self.colors = {
            "black": QtGui.QColor(0, 0, 0),
            "red": QtGui.QColor(255, 0, 0),
            "green": QtGui.QColor(0, 128, 0),
            "blue": QtGui.QColor(0, 0, 128)
        }

        #Updating Log Section
        self.updateLogScreen("Application Initiated!", self.colors["black"])
        self.leftLayout.addSpacing(15)

        devLabel = QLabel()
        devLabel.setText("Developed By: M. Raaid Khan")
        devLabel.setAlignment(Qt.AlignLeft)
        devLabel.setStyleSheet("font-size: 15px; color: green;")
        devLabel.setFixedSize(330,25)
        self.leftLayout.addWidget(devLabel)

        # Adding Stretch at Bottom to make everything Visible
        self.leftLayout.addStretch()


        # <<--Left Layout Design Complete-->>  


        # <<--Desining Mid Layout-->>  

        # <Top Bar>
        topBar = QFrame()
        topBar.setFixedHeight(100)
        self.topBarLayout = QHBoxLayout()
        topBar.setLayout(self.topBarLayout)
        self.topBarLayout.setContentsMargins(10, 0, 10, 0)
        self.topBarLayout.setSpacing(10)
        topBar.setFrameStyle(QFrame.Panel)

        self.topBarLayout.addSpacing(30)
        # <-- Start Button-->
        self.startButton = QPushButton("  Start Detection")
        self.startButton.setToolTip("Start Object Detection on Live Camera Feed")
        # self.btn1.setText("Home")
        self.startButton.setStyleSheet("""
            QPushButton {
                padding: 10px 20px;  /* Top/Bottom and Left/Right padding */
                font-size: 18px;      /* Text font size */
                background-color: #FDFDFD;  /* Button background color */
                color: black;         /* Text color */
                border-radius: 10px;  /* Curved corners */
                border: 1px solid #BDBDBD; /* Button border */
            }
            QPushButton:hover {
                background-color: #F5F5F5;  /* Darker shade when hovered */
            }
        """)
        self.startButton.setFixedSize(200, 60)
        # self.btn1.setAlignment(Qt.AlignCenter)
        self.startButton.setContentsMargins(0, 0, 0, 0)
        self.startButton.setIcon(QtGui.QIcon('./Data/Start.png'))
        self.topBarLayout.addWidget(self.startButton)
        self.topBarLayout.addSpacing(430)
        self.startButton.clicked.connect(self.toggle_video_feed)


        # <-- Camera Selector-->
        # Create a label
        Cameralabel = QLabel('Camera:')
        Cameralabel.setStyleSheet("font-size: 22px;")
        # Create a dropdown
        Cameradropdown = QComboBox(self)
        Cameradropdown.addItems(['Device 0', 'Device 1'])
        Cameradropdown.setCurrentIndex(0)
        Cameradropdown.setFixedSize(150, 30)
        Cameradropdown.setStyleSheet("font-size: 18px; text-align: center;")

        def on_device_changed():
            global device_number

            selected_device = Cameradropdown.currentText()

            if selected_device == 'Device 0':
                device_number = 0
            elif selected_device == 'Device 1':
                device_number = 1

        Cameradropdown.currentIndexChanged.connect(on_device_changed)

        # Create a horizontal layout
        Cameralayout = QHBoxLayout()
        Cameralayout.addWidget(Cameralabel)
        Cameralayout.addWidget(Cameradropdown)
        self.topBarLayout.addLayout(Cameralayout)
        self.topBarLayout.addSpacing(50)

        self.midLayout.addWidget(topBar)
        self.midLayout.addSpacing(10)
        # <Top Bar Completed>

        # <--Video Area in Mid Layout-->

        # Create a QLabel to display the camera feed
        self.cameraFeed = QLabel(self)
        self.cameraFeed.setFixedSize(1000, 600)
        self.cameraFeed.setStyleSheet("border: 1px solid black;")
        self.cameraFeed.setAlignment(Qt.AlignCenter)

        self.midLayout.addWidget(self.cameraFeed)


        # Adding Stretch at Bottom to make everything Visible
        self.midLayout.addStretch()

        # Adding Borders to Frames
        self.frameLeft.setFrameStyle(QFrame.Panel | QFrame.Raised)
        self.frameMid.setFrameStyle(QFrame.Panel | QFrame.Raised)
        self.frameRight.setFrameStyle(QFrame.Panel | QFrame.Raised)

        # Set the layout for the main window
        self.main_layout.addWidget(self.frameLeft)
        self.main_layout.addWidget(self.frameMid)
        self.main_layout.addWidget(self.frameRight)
        self.setLayout(self.main_layout)

    def toggle_video_feed(self):
        if self.is_video_running:
            # Stop the video feed
            self.camera_thread.stop_capture()
            self.camera_thread.release()
            self.startButton.setText('  Start Detection')
            self.startButton.setIcon(QtGui.QIcon('./Data/Start.png'))
            self.is_video_running = False
            self.camera_thread.quit()
            self.camera_thread.wait()
            self.cameraFeed.clear()
            self.cameraFeed.setText('Camera Feed Stopped')
            
        else:
            # Start the video feed
            self.camera_thread = CameraThread()
            self.camera_thread.frame_signal.connect(self.update_frame)
            self.camera_thread.start_capture()
            self.startButton.setText('  Stop Detection')
            self.startButton.setIcon(QtGui.QIcon('./Data/Stop.png'))
            self.is_video_running = True

    def update_frame(self, frame):
        # Convert the frame from BGR (OpenCV format) to RGB (PyQt format)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert the frame to QImage
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Convert QImage to QPixmap and set it to the label
        pixmap = QPixmap.fromImage(qimg)
        self.cameraFeed.setPixmap(pixmap)

    def closeEvent(self, event):
        # Release the camera and close the application
        self.camera_thread.stop_capture()
        self.camera_thread.release()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    myApp = MyApp()

    myApp.showMaximized()
    
    try:
        sys.exit(app.exec_())
    except SystemExit:
        print('Closing Window...')