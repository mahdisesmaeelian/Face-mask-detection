from PySide6.QtWidgets import *
from PySide6.QtUiTools import *
from PySide6.QtCore import *
from PySide6.QtGui import *
import cv2
import numpy as np
from tensorflow.keras.models import load_model

model=load_model('Face Mask Dataset.h5')

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        loader = QUiLoader()
        self.ui = loader.load("Face Mask Inference.ui", None)
        self.ui.show()
        self.ui.setWindowTitle("Face Mask Detector")
        self.Camera()

    def Camera(self):
        self.cap = cv2.VideoCapture(0)
        
        while True:
            ret, image = self.cap.read()

            if ret==False :
                break

            height, width, channel = image.shape
            step = channel * width
            qImg = QImage(image.data, width, height, step, QImage.Format_BGR888)
            self.ui.labelCam.setPixmap(QPixmap.fromImage(qImg))
            cv2.waitKey(1)

            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224,224))
            img = img/255
            img = img.reshape(1, 224,224, 3)
            pred = model.predict(img)
            result = np.argmax(pred)
            
            if result==0:
                self.ui.label.setText('With Mask')
            elif result==1:
                self.ui.label.setText('WithOut Mask')  

            if cv2.waitKey(1) == ord("q"):
                break
            
app = QApplication()
window = MainWindow()
app.exec()