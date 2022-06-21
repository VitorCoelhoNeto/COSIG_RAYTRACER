import base64
from copy import deepcopy
from tracemalloc import start
from turtle import distance
from OpenGL import GL, GLUT
import OpenGL.GLU as GLU
from PyQt6.QtCore import QSize, Qt, QRect
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QSlider, QLabel, QFileDialog, QTextEdit, QMessageBox, QLineEdit
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from app import generate_scene_objects, preliminar_calculations, parser
import PIL.Image as Image
from PIL import ImageOps
import io

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import time



# Subclass QMainWindow to customize your application's main window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.recursionLevel=1
        self.parser=None
        #self.startTime = time.time()
        self.setWindowTitle("COSIG Ray Tracer")
        self.setFixedSize(QSize(450, 530))

        self.imageHeight=0
        self.imageWidth=0
        self.pixelScale=0
        self.pixels=[]
        self.counter=0
        self.pixelListMain = list()
        
        layoutVertical = QVBoxLayout()
        self.opengl_window = GLWidget()
        
        layoutHorizontal = QHBoxLayout()
        layoutHorizontal2 = QHBoxLayout()
        layoutHorizontal3 = QHBoxLayout()
        layoutHorizontal4 = QHBoxLayout()
        loadButton = QPushButton("Load")
        loadButton.clicked.connect(self.loadFile)
        saveButton = QPushButton("Save")
        saveButton.clicked.connect(self.saveFile)
        startButton = QPushButton("Start")
        startButton.clicked.connect(self.startRaytracing)
        textBoxRec = QLabel("Rec level: ")
        self.textBoxInput = QLineEdit()
        self.textBoxInput.resize(20, 20)
        self.textBoxInput.setText("2")
        
        labelDistance = QLabel("Distance: ")
        self.distanceTextBox = QLineEdit()
        self.distanceTextBox.resize(20, 20)
        self.distanceTextBox.setText("30")
        labelFov = QLabel("FOV: ")
        self.fovTextBox = QLineEdit()
        self.fovTextBox.resize(20, 20)
        self.fovTextBox.setText("30")
        self.labelElapsedTime = QLabel("Elapsed Time: ---                                                                                                            ")
        self.labelElapsedTime.setGeometry(QRect(10, 10, 10, 10))

        layoutHorizontal.addWidget(loadButton)
        layoutHorizontal.addWidget(startButton)
        layoutHorizontal.addWidget(saveButton)
        layoutHorizontal2.addWidget(textBoxRec)
        layoutHorizontal2.addWidget(self.textBoxInput)
        layoutHorizontal3.addWidget(labelDistance)
        layoutHorizontal3.addWidget(self.distanceTextBox)
        layoutHorizontal3.addWidget(labelFov)
        layoutHorizontal3.addWidget(self.fovTextBox)
        layoutHorizontal2.addWidget(self.labelElapsedTime)


        #self.opengl_window.setGeometry(QRect(200, 200, 0, 0 ))
        #self.opengl_window.setObjectName("OpenGL Window")
        layoutVertical.addWidget(self.opengl_window)
        layoutVertical.addLayout(layoutHorizontal)
        layoutVertical.addLayout(layoutHorizontal2)
        layoutVertical.addLayout(layoutHorizontal3)
        #layoutVertical.addLayout(layoutHorizontal4)

        # Set the central widget of the Window.
        widget = QWidget()
        widget.setLayout(layoutVertical)
        self.setCentralWidget(widget)


    def loadFile(self):
        path = QFileDialog.getOpenFileName(self, 'Open a file', '',
                                            'Scene Files (*.txt)')
        if path != ('', ''):
            print("File path : " + path[0])
        image_path = path[0]

        self.startTime = time.time()
        # Parse the loaded file and fill spin boxes
        imageContents = parser(image_path)
        
        # Calculations for the ray paths as well as sceneObjects list
        LIST_CAM = 0
        LIST_IMG = 1
        LIST_SPH = 2
        LIST_LIG = 3
        LIST_BOX = 4
        LIST_FLO = 5
        LIST_PYR = 6
        LIST_DON = 7
        sceneObjects = generate_scene_objects(imageContents, self.fovTextBox.text(), self.distanceTextBox.text())
        pixelList = preliminar_calculations(sceneObjects[LIST_CAM], sceneObjects[LIST_IMG], sceneObjects, int(self.textBoxInput.text()))
        
        for pixel in pixelList:
            self.pixelListMain.append([int(pixel.red), int(pixel.green), int(pixel.blue)])
        self.pixelListMain.reverse()
        self.endTime = time.time()

    def startRaytracing(self):
        print("start button clicked")
        self.showFinalImage()

    def showFinalImage(self):
        #arrayOfArrays=[ [int(x.red), int(x.green), int(x.blue)] for y in self.pixelListMain for x in y ]
        matrixDecomposition = [item for sublist in self.pixelListMain for item in sublist]
        self.matrixDecomposition = deepcopy(matrixDecomposition)
        print("will now paint the pixels")
        self.opengl_window.imageWidth=int(200)
        self.opengl_window.imageHeight=int(200)
        self.opengl_window.imageData=matrixDecomposition
        self.opengl_window.paintGL()

        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        data = glReadPixels(0, 0, 200, 200, GL_RGBA, GL_UNSIGNED_BYTE)
        image = Image.frombytes("RGBA", (200, 200), data)
        image = ImageOps.flip(image)
        image = ImageOps.mirror(image)
        self.finalImage = image
        #endTime = time.time()
        finalTime = self.endTime - self.startTime
        self.labelElapsedTime.setText(f"Elapsed time: {round(float(finalTime), 2)} s                                                                                                            ")
    
    def saveFile(self):
        self.finalImage.save('finalImage.png', 'PNG')


class GLWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        self.parent = parent
        QOpenGLWidget.__init__(self, parent)
        self.imageWidth=0
        self.imageHeight=0
        self.imageData=[]

    def initializeGL(self):
        #self.qglClearColor(QtGui.QColor(0, 0, 255))    # initialize the screen to blue
        GL.glClearColor(255, 255, 255, 0.5) # azul com transparencia
        GL.glEnable(GL.GL_DEPTH_TEST)

    def resizeGL(self, width, height):
        GL.glViewport(0, 0, width, height)
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        aspect = width / float(height)
    
        GLU.gluPerspective(45.0, aspect, 1.0, 100.0)
        GL.glMatrixMode(GL.GL_MODELVIEW)

    def paintGL(self):
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glDrawPixels(self.imageWidth, self.imageHeight, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, (GL.GLubyte * len(self.imageData))(*self.imageData))
        self.update()
