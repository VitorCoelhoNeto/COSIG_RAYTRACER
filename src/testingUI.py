from OpenGL import GL
import OpenGL.GLU as GLU
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QSlider, QLabel, QFileDialog
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from app import generate_scene_objects, preliminar_calculations, parser


# Subclass QMainWindow to customize your application's main window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.recursionLevel=1
        self.parser=None

        self.setWindowTitle("COSIG Ray Tracer")
        self.setFixedSize(QSize(800, 700))

        self.imageHeight=0
        self.imageWidth=0
        self.pixelScale=0
        self.pixels=[]
        self.counter=0
        self.pixelListMain = list()
        
        layoutVertical = QVBoxLayout()
        self.opengl_window = GLWidget()
        
        layoutHorizontal = QHBoxLayout()
        loadButton = QPushButton("Load")
        loadButton.clicked.connect(self.loadFile)
        startButton = QPushButton("Start")
        startButton.clicked.connect(self.startRaytracing)
        self.labelSliderValue = QLabel("Recursion level: 1")

        sliderLabelLayoutHorizontal = QHBoxLayout()
        self.slider = QSlider(orientation=Qt.Orientation.Horizontal)
        
        sliderLabelLayoutHorizontal.addWidget(self.labelSliderValue)
        sliderLabelLayoutHorizontal.addWidget(self.slider)

        layoutHorizontal.addWidget(loadButton)
        layoutHorizontal.addWidget(startButton)
        layoutHorizontal.addLayout(sliderLabelLayoutHorizontal)
        
        layoutVertical.addWidget(self.opengl_window)
        layoutVertical.addLayout(layoutHorizontal)

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
        sceneObjects = generate_scene_objects(imageContents)
        pixelList, rayList = preliminar_calculations(sceneObjects[LIST_CAM], sceneObjects[LIST_IMG], sceneObjects)
        
        for pixel in pixelList:
            self.pixelListMain.append([int(pixel.red), int(pixel.green), int(pixel.blue)])

    def startRaytracing(self):
        print("start button clicked")
        self.showFinalImage()

    def showFinalImage(self):
        #arrayOfArrays=[ [int(x.red), int(x.green), int(x.blue)] for y in self.pixelListMain for x in y ]
        matrixDecomposition = [item for sublist in self.pixelListMain for item in sublist]
        print("will now paint the pixels")
        self.opengl_window.imageWidth=int(200)
        self.opengl_window.imageHeight=int(200)
        self.opengl_window.imageData=matrixDecomposition
        self.opengl_window.paintGL()


class GLWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        self.parent = parent
        QOpenGLWidget.__init__(self, parent)
        self.imageWidth=0
        self.imageHeight=0
        self.imageData=[]

    def initializeGL(self):
        #self.qglClearColor(QtGui.QColor(0, 0, 255))    # initialize the screen to blue
        GL.glClearColor(0, 0, 255, 0.5) # azul com transparencia
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