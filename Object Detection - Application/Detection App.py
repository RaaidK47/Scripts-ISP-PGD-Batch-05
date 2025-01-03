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
from ultralytics import YOLO
import torch
from PyQt5.QtWidgets import QSlider
import pandas as pd

print(torch.cuda.is_available())

global yoloVersion
yoloVersion = "YOLOv9"

global version_num
version_num = 9

global camera_number
camera_number = 0

global yoloModel
yoloModel = "yolov9t.pt"

global device_selection
device_selection = "GPU"

global unique_classes
unique_classes = set()

global preprocess_list
preprocess_list = []

global inference_list
inference_list = []

global postprocess_list
postprocess_list = []

global conf_val
global rect_thickness
global text_thickness
global text_scale

conf_val = 0.5
rect_thickness = 1
text_thickness = 1
text_scale = 0.5

class CameraThread(QThread):
    # Signal to update the GUI with new frame
    frame_signal = pyqtSignal(tuple)

    def __init__(self):
        global camera_number
        super().__init__()
        self.is_running = False
        self.cap = cv2.VideoCapture(camera_number)  # Initialize the camera


    def set_model(self, model_name):
        """Slot to update the model name"""
        self.model_name = model_name
        print(f"Model Set to: {self.model_name}")

    def run(self):
        global yoloModel
        global device_selection
        global unique_classes
        global preprocess_list
        global inference_list
        global postprocess_list

        # Resetting "Unique Classes on Restart of Detection"
        unique_classes = set()

        # Restting Process Times List on Restart of Detection
        preprocess_list = []
        inference_list = []
        postprocess_list = []

        model = YOLO(f'{yoloModel.lower()}')

        if device_selection == 'GPU':
            if torch.cuda.is_available():
                print("CUDA is available; Using GPU.")
                model.to(torch.device('cuda'))
                
        elif device_selection == 'CPU':
            model.to(torch.device('cpu'))
            print("Using CPU.")
       
        global conf_val

        while self.is_running:
            ret, frame = self.cap.read()

            if ret:
                frame = cv2.flip(frame, 1)  # Flip the frame horizontally
                results = model.predict(frame, conf=conf_val)
                self.frame_signal.emit((frame, results))
            else:
                print("Failed to capture frame.")

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

    def updatePredictionLogScreen(self, text, color):
        self.predLogs.setTextColor(color)
        self.predLogs.insertPlainText(text+"\n")
        self.predLogs.verticalScrollBar().setSliderDown(1)
        self.predLogs.moveCursor(QtGui.QTextCursor.End)
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
            self.Modeldropdown.clear()
            if yoloVersion == 'YOLOv8':
                self.Modeldropdown.addItems(['YOLOv8n', 'YOLOv8s', 'YOLOv8m', 'YOLOv8l', 'YOLOv8x'])
                self.Modeldropdown.setCurrentIndex(1)
            elif yoloVersion == 'YOLOv9':
                self.Modeldropdown.addItems(['YOLOv9t', 'YOLOv9s', 'YOLOv9m', 'YOLOv9c', 'YOLOv9e'])
                self.Modeldropdown.setCurrentIndex(1)
            elif yoloVersion == 'YOLOv10':
                self.Modeldropdown.addItems(['YOLOv10n', 'YOLOv10s', 'YOLOv10m', 'YOLOv10l', 'YOLOv10x'])
                self.Modeldropdown.setCurrentIndex(1)
            elif yoloVersion == 'YOLOv11':
                self.Modeldropdown.addItems(['YOLO11n', 'YOLO11s', 'YOLO11m', 'YOLO11l', 'YOLO11x'])
                self.Modeldropdown.setCurrentIndex(1)

            self.updateLogScreen(f"{yoloVersion} Version Selected...", self.colors["green"])

        # Connect the currentTextChanged signal to the slot
        YOLOdropdown.currentIndexChanged.connect(on_version_changed)

        # <-- YOLO Model Selector-->
        # Create a label
        Modellabel = QLabel('YOLO Model:', self)
        Modellabel.setStyleSheet("font-size: 20px; font-weight: bold;")
        # Create a dropdown
        self.Modeldropdown = QComboBox(self)
        self.Modeldropdown.addItems(['YOLOv9t', 'YOLOv9s', 'YOLOv9m', 'YOLOv9c', 'YOLOv9e'])
        self.Modeldropdown.setStyleSheet("font-size: 18px;")
        # Create a horizontal layout
        Modellayout = QHBoxLayout()
        Modellayout.addWidget(Modellabel)
        Modellayout.addWidget(self.Modeldropdown)
        self.leftLayout.addLayout(Modellayout)
        self.leftLayout.addSpacing(30)

        def updateModel():
            global yoloModel
            yoloModel = self.Modeldropdown.currentText()
            print("YOLO Model set to "+ yoloModel)

        # Connect the dropdown change to the thread's model signal
        self.Modeldropdown.currentTextChanged.connect(updateModel)

        

        # <-- Device Selector-->
        # Create a label
        Devicelabel = QLabel('Device:', self)
        Devicelabel.setStyleSheet("font-size: 20px; font-weight: bold;")
        # Create a dropdown
        self.Devicedropdown = QComboBox(self)
        self.Devicedropdown.addItems(['CPU', 'GPU'])
        self.Devicedropdown.setCurrentIndex(1)
        self.Devicedropdown.setStyleSheet("font-size: 18px;")
        # Create a horizontal layout
        Devicelayout = QHBoxLayout()
        Devicelayout.addWidget(Devicelabel)
        Devicelayout.addWidget(self.Devicedropdown)
        self.leftLayout.addLayout(Devicelayout)
        self.leftLayout.addSpacing(30)

        def updateDevice():
            global device_selection
            device_selection = self.Devicedropdown.currentText()
            self.updateLogScreen(f"\"{device_selection}\" Selected as Active Device", self.colors["blue"])

        self.Devicedropdown.currentTextChanged.connect(updateDevice)

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
        Cameradropdown.addItems(['Camera 0', 'Camera 1'])
        Cameradropdown.setCurrentIndex(0)
        Cameradropdown.setFixedSize(150, 30)
        Cameradropdown.setStyleSheet("font-size: 18px; text-align: center;")

        def on_device_changed():
            global camera_number

            selected_camera = Cameradropdown.currentText()

            if selected_camera == 'Camera 0':
                camera_number = 0
                self.updateLogScreen("Camera 0 Selected", self.colors["blue"])
            elif selected_camera == 'Camera 1':
                camera_number = 1
                self.updateLogScreen("Camera 1 Selected", self.colors["blue"])


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
        self.cameraFeed.setFixedSize(1000, 520)
        self.cameraFeed.setStyleSheet("border: 1px solid black;font-size: 18px;")
        self.cameraFeed.setAlignment(Qt.AlignCenter)
        self.cameraFeed.setText("Feed Not Detected")

        self.midLayout.addWidget(self.cameraFeed)


        #<-- Prediction Logs-->
        predlogLabel = QLabel()
        predlogLabel.setText("Prediction Logs")
        predlogLabel.setAlignment(Qt.AlignCenter)
        # logLabel.setFixedSize(330,35)
        predlogLabel.setFont(QtGui.QFont("ROLNER", 11))
        predlogLabel.setContentsMargins(0,5,0,0)
        self.midLayout.addWidget(predlogLabel)
        self.midLayout.addSpacing(10)

        self.predLogs = QTextEdit()
        self.predLogs.setReadOnly(True)
        self.predLogs.setFixedSize(1000,260)
        self.predLogs.setFont(QtGui.QFont("arial-unicode-ms", 9))
        self.midLayout.addWidget(self.predLogs)

        # Adding Stretch at Bottom to make everything Visible
        self.midLayout.addStretch()

        # <<--Mid Layout Design Completed-->>  


        # <<--Designing Right Layout-->>  


        self.rightLayout.addSpacing(30)

        # Adding a Heading of "Parameters"
        self.heading = QLabel("Parameters")
        self.heading.setStyleSheet("font-size: 25px; font-weight: bold;")
        self.heading.setAlignment(Qt.AlignCenter)
        self.rightLayout.addWidget(self.heading)
        self.rightLayout.addSpacing(50)

        # Create a label for Confidence
        self.confLabel = QLabel('Confidence:   ')
        self.confLabel.setStyleSheet("font-size: 20px")

        # Create the slider for Confidence
        self.confSlider = QSlider(Qt.Horizontal)
        self.confSlider.setMinimum(0)
        self.confSlider.setMaximum(100)
        self.confSlider.setValue(50)  # Default value is 50
        self.confSlider.setTickPosition(QSlider.TicksBelow)
        self.confSlider.setTickInterval(10)
        self.confSlider.setFixedSize(200, 30)

        # Create a label to show the current value of the slider
        self.confValueLabel = QLabel('50')  # Display the initial value of the slider
        self.confValueLabel.setStyleSheet("font-size: 18px")

        # Layout to hold the components
        self.confHLayout = QHBoxLayout()
        self.confHLayout.addWidget(self.confLabel)
        self.confHLayout.addSpacing(25)
        self.confHLayout.addWidget(self.confSlider)
        self.confHLayout.addSpacing(10)
        self.confHLayout.addWidget(self.confValueLabel)

        self.rightLayout.addLayout(self.confHLayout)

        # Connect the slider's valueChanged signal to the update method
        self.confSlider.valueChanged.connect(self.update_conf_value)

        self.rightLayout.addSpacing(50)

        

        # Label and Slider for Rectangle Thickness
        self.rectLabel = QLabel('Rect Thickness:')
        self.rectLabel.setStyleSheet("font-size: 20px")
        self.rectSlider = QSlider(Qt.Horizontal)
        self.rectSlider.setMinimum(1)
        self.rectSlider.setMaximum(5)
        self.rectSlider.setValue(1)
        self.rectSlider.setTickPosition(QSlider.TicksBelow)
        self.rectSlider.setTickInterval(1)
        self.rectSlider.setFixedSize(200, 30)
        self.rectValueLabel = QLabel('1')
        self.rectValueLabel.setStyleSheet("font-size: 18px")
        self.rectHLayout = QHBoxLayout()
        self.rectHLayout.addWidget(self.rectLabel)
        self.rectHLayout.addSpacing(10)
        self.rectHLayout.addWidget(self.rectSlider)
        self.rectHLayout.addSpacing(10)
        self.rectHLayout.addWidget(self.rectValueLabel)
        self.rightLayout.addLayout(self.rectHLayout)

        self.step_size = 1  # Custom step size
        # Changing Slider Default Step Size (i.e 10) to 1
        self.rectSlider.mousePressEvent = self.rect_slider_mousePressEvent

        self.rectSlider.valueChanged.connect(self.update_rect_value)

        self.rightLayout.addSpacing(50)

        # Label and Slider for Text Thickness
        self.textLabel = QLabel('Text Thickness:')
        self.textLabel.setStyleSheet("font-size: 20px")
        self.textSlider = QSlider(Qt.Horizontal)
        self.textSlider.setMinimum(1)
        self.textSlider.setMaximum(5)
        self.textSlider.setValue(1)
        self.textSlider.setTickPosition(QSlider.TicksBelow)
        self.textSlider.setTickInterval(1)
        self.textSlider.setFixedSize(200, 30)
        self.textValueLabel = QLabel('1')
        self.textValueLabel.setStyleSheet("font-size: 18px")
        self.textHLayout = QHBoxLayout()
        self.textHLayout.addWidget(self.textLabel)
        self.textHLayout.addSpacing(10)
        self.textHLayout.addWidget(self.textSlider)
        self.textHLayout.addSpacing(10)
        self.textHLayout.addWidget(self.textValueLabel)
        self.rightLayout.addLayout(self.textHLayout)
        self.textSlider.valueChanged.connect(self.update_text_value)
        # Changing Slider Default Step Size (i.e 10) to 1
        self.textSlider.mousePressEvent = self.text_slider_mousePressEvent

        self.rightLayout.addSpacing(50)

        # Label and Slider for Text Size
        self.textSizeLabel = QLabel('Text Size:        ')
        self.textSizeLabel.setStyleSheet("font-size: 20px")
        self.textSizeSlider = QSlider(Qt.Horizontal)
        self.textSizeSlider.setMinimum(1)
        self.textSizeSlider.setMaximum(6)
        self.textSizeSlider.setValue(1)
        self.textSizeSlider.setTickPosition(QSlider.TicksBelow)
        self.textSizeSlider.setTickInterval(1)
        self.textSizeSlider.setFixedSize(200, 30)
        self.textSizeValueLabel = QLabel('0.5')
        self.textSizeValueLabel.setStyleSheet("font-size: 18px")
        self.textSizeHLayout = QHBoxLayout()
        self.textSizeHLayout.addWidget(self.textSizeLabel)
        self.textSizeHLayout.addSpacing(10)
        self.textSizeHLayout.addWidget(self.textSizeSlider)
        self.textSizeHLayout.addSpacing(10)
        self.textSizeHLayout.addWidget(self.textSizeValueLabel)
        self.rightLayout.addLayout(self.textSizeHLayout)

        self.textSizeSlider.valueChanged.connect(self.update_text_size_value)
        # Changing Slider Default Step Size (i.e 10) to 1
        self.textSizeSlider.mousePressEvent = self.textsize_slider_mousePressEvent

        self.rightLayout.addSpacing(50)
        # Results Sections
        self.result = QLabel("Results")
        self.result.setStyleSheet("font-size: 25px; font-weight: bold;")
        self.result.setAlignment(Qt.AlignCenter)
        self.rightLayout.addWidget(self.result)
        self.rightLayout.addSpacing(30)

        # A dynamic results label that shows all the results of Object Detection Results
        self.classesLabel = QLabel("Classes Detected")
        self.classesLabel.setStyleSheet("font-size: 20px; font-weight: bold;")
        self.classesLabel.setAlignment(Qt.AlignLeft)
        self.rightLayout.addWidget(self.classesLabel)
        self.rightLayout.addSpacing(20)


        self.clasees_detected = QLabel()
        self.clasees_detected.setStyleSheet("font-size: 17px;")
        self.clasees_detected.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        self.rightLayout.addWidget(self.clasees_detected)
        self.clasees_detected.setText('----')
        self.clasees_detected.setFixedHeight(200)

        # Results Time 
        self.resultTimeHeading = QLabel("Time Taken")
        self.resultTimeHeading.setStyleSheet("font-size: 20px; font-weight: bold;")
        self.resultTimeHeading.setAlignment(Qt.AlignLeft)
        self.rightLayout.addWidget(self.resultTimeHeading)
        # self.rightLayout.addSpacing(20)

        self.resultstime = QLabel()
        self.resultstime.setStyleSheet("font-size: 17px;")
        self.resultstime.setAlignment(Qt.AlignLeft)
        self.rightLayout.addWidget(self.resultstime)
        self.resultstime.setText("""
PreProcess   -> Min = --- ms  : Max = --- ms\n
Inference     -> Min = --- ms  : Max = --- ms\n
PostProcess -> Min = --- ms  : Max = --- ms\n""")

        # Adding Stretch at Bottom to make everything Visible
        self.rightLayout.addStretch()

        # Adding Borders to Frames
        self.frameLeft.setFrameStyle(QFrame.Panel | QFrame.Raised)
        self.frameMid.setFrameStyle(QFrame.Panel | QFrame.Raised)
        self.frameRight.setFrameStyle(QFrame.Panel | QFrame.Raised)

        # Set the layout for the main window
        self.main_layout.addWidget(self.frameLeft)
        self.main_layout.addWidget(self.frameMid)
        self.main_layout.addWidget(self.frameRight)
        self.setLayout(self.main_layout)

    def rect_slider_mousePressEvent(self, event):
        # Calculate the current slider value based on the click position
        if event.button() == Qt.LeftButton:
            x_position = event.pos().x()
            slider_width = self.rectSlider.width()
            clicked_value = self.rectSlider.minimum() + (
                (self.rectSlider.maximum() - self.rectSlider.minimum()) * x_position / slider_width
            )

            # Adjust the value by the desired step size
            current_value = self.rectSlider.value()
            if clicked_value > current_value:
                new_value = min(self.rectSlider.maximum(), current_value + self.step_size)
            else:
                new_value = max(self.rectSlider.minimum(), current_value - self.step_size)

            self.rectSlider.setValue(int(new_value))  # Update the slider value

        # Call the base class implementation for other mouse events
        super(QSlider, self.rectSlider).mousePressEvent(event)

    def text_slider_mousePressEvent(self, event):
        # Calculate the current slider value based on the click position
        if event.button() == Qt.LeftButton:
            x_position = event.pos().x()
            slider_width = self.textSlider.width()
            clicked_value = self.textSlider.minimum() + (
                (self.textSlider.maximum() - self.textSlider.minimum()) * x_position / slider_width
            )

            # Adjust the value by the desired step size
            current_value = self.textSlider.value()
            if clicked_value > current_value:
                new_value = min(self.textSlider.maximum(), current_value + self.step_size)
            else:
                new_value = max(self.textSlider.minimum(), current_value - self.step_size)

            self.textSlider.setValue(int(new_value))  # Update the slider value

        # Call the base class implementation for other mouse events
        super(QSlider, self.textSlider).mousePressEvent(event)

    def textsize_slider_mousePressEvent(self, event):
        # Calculate the current slider value based on the click position
        if event.button() == Qt.LeftButton:
            x_position = event.pos().x()
            slider_width = self.textSizeSlider.width()
            clicked_value = self.textSizeSlider.minimum() + (
                (self.textSizeSlider.maximum() - self.textSizeSlider.minimum()) * x_position / slider_width
            )

            # Adjust the value by the desired step size
            current_value = self.textSizeSlider.value()
            if clicked_value > current_value:
                new_value = min(self.textSizeSlider.maximum(), current_value + self.step_size)
            else:
                new_value = max(self.textSizeSlider.minimum(), current_value - self.step_size)

            self.textSizeSlider.setValue(int(new_value))  # Update the slider value

        # Call the base class implementation for other mouse events
        super(QSlider, self.textSizeSlider).mousePressEvent(event)

    def update_conf_value(self):
        global conf_val
        # Update the label with the current slider value
        current_value = self.confSlider.value()
        self.confValueLabel.setText(str(current_value))
        conf_val = int(current_value) / 100


    def update_rect_value(self):
        # Update the label with the current slider value
        current_value = self.rectSlider.value()
        self.rectValueLabel.setText(str(current_value))
        global rect_thickness
        rect_thickness = int(current_value)

    def update_text_value(self):
        # Update the label with the current slider value
        current_value = self.textSlider.value()
        self.textValueLabel.setText(str(current_value))
        global text_thickness
        text_thickness = int(current_value)

    def update_text_size_value(self):
        # Update the label with the current slider value
        current_value = self.textSizeSlider.value()
        self.textSizeValueLabel.setText(str(current_value/2))
        global text_scale
        text_scale = float(current_value/2)

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
            self.updateLogScreen("Detection Stoped...", self.colors["red"])
            
        else:
            # Start the video feed
            self.camera_thread = CameraThread()
            self.camera_thread.frame_signal.connect(self.update_frame)
            self.camera_thread.start_capture()
            self.startButton.setText('  Stop Detection')
            self.startButton.setIcon(QtGui.QIcon('./Data/Stop.png'))
            self.is_video_running = True
            self.updateLogScreen("Detection Started...", self.colors["green"])
            self.updateLogScreen(f"Confidence Level: {conf_val*100}%\nRectangle Thickness: {rect_thickness}\nText Thickness: {text_thickness}\nText Size: {text_scale}",
                                  self.colors["blue"])


    def predict(chosen_model, img, classes=[], conf=0.5):
        # Check if classes are specified for filtering
        # if classes:
        #     results = chosen_model.predict(img, classes=classes, conf=conf)
        # else:
        results = chosen_model.predict(img, conf=conf)

        return results

    def update_frame(self, data):
        frame, results = data

        global conf_val
        global rect_thickness
        global text_thickness
        global text_scale
        

        # Convert the frame from BGR (OpenCV format) to RGB (PyQt format)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Assuming you are working with a single image result, access the first element
        result = results[0]

        # Get speed metrics from the results
        speed = result.speed  # Dictionary containing speed metrics
        preprocess_time = speed.get('preprocess', 'N/A')
        inference_time = speed.get('inference', 'N/A')
        postprocess_time = speed.get('postprocess', 'N/A')

        # Get bounding boxes, labels, and confidence scores
        # boxes = result.boxes.xywh  # Bounding boxes (x, y, width, height)
        confidences = result.boxes.conf  # Confidence scores
        labels = result.names  # Class names


        global unique_classes
        # Get Unique Classes from Results
        for label in result.boxes.cls:
            unique_classes.add(labels[int(label)])
            # print(unique_classes)


        unique_classes_str =  ", ".join(word.capitalize() for word in unique_classes)
        self.clasees_detected.setText(str(unique_classes_str))

        global preprocess_list
        global inference_list
        global postprocess_list

        # Not Adding time = 0 ms
        if preprocess_time != 0:
            preprocess_list.append(preprocess_time)
        
        if inference_time != 0:
            inference_list.append(inference_time)
        
        if postprocess_time != 0:
            postprocess_list.append(postprocess_time)

        # Check that none of the three lists is empty
        all_not_empty = all(len(lst) > 0 for lst in [preprocess_list, inference_list, postprocess_list])

        if all_not_empty:
            self.resultstime.setText(f"""
PreProcess   -> Min = {min(preprocess_list):.2f} ms  : Max = {max(preprocess_list):.2f} ms\n
Inference     -> Min = {min(inference_list):.2f} ms  : Max = {max(inference_list):.2f} ms\n
PostProcess -> Min = {min(postprocess_list):.2f} ms  : Max = {max(postprocess_list):.2f} ms\n""")
            
        else:
            self.resultstime.setText("""
PreProcess   -> Min = --- ms  : Max = --- ms\n
Inference     -> Min = --- ms  : Max = --- ms\n
PostProcess -> Min = --- ms  : Max = --- ms\n""")


        if result:
            # Format the results as a string
            result_str = ""
            for i, label in enumerate(result.boxes.cls):
                label_name = labels[int(label)]  # Get the class name
                confidence = confidences[i]  # Get the confidence score

                # Adding the 2nd Onwards results on New Line
                if (i == 0):
                    result_str += f"Detected <{label_name.upper()}> with {confidence*100:.0f}% Confidence"
                else:
                    result_str += f"\nDetected <{label_name.upper()}> with {confidence*100:.0f}% Confidence"
            
            self.updatePredictionLogScreen(result_str, self.colors["blue"])
        
            speed_results = f"Preprocess: {preprocess_time:.2f}ms, Inference: {inference_time:.2f}ms, Postprocess: {postprocess_time:.2f}ms"
            self.updatePredictionLogScreen(speed_results, self.colors["green"])
            self.updatePredictionLogScreen("------------------------------------------------------------------------------------------------\n",
                                            self.colors["black"])

        # Show the DataFrame
        # print(predictions_df)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                class_name = result.names[int(box.cls[0])] # Class name
                confidence = float(box.conf[0])  # Confidence score
                label = f"{class_name}  {confidence*100:.2f}%"  # Combine class name and confidence
                cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), (255, 0, 0), rect_thickness)
                cv2.putText(rgb_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255, 0, 0), text_thickness)

        # Convert to QImage and display
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.cameraFeed.setPixmap(pixmap)

        self.cameraFeed.setPixmap(QPixmap.fromImage(qimg))
        

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