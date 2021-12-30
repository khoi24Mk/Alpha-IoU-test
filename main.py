import argparse
import concurrent
import sys
from os import walk

from pathlib import Path
from PyQt5 import QtCore
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.uic.properties import QtGui

from loading_screen import Ui_LoadingWindow
# from main_stuff import Ui_MainWindow
from process_screen import Ui_MainWindow
from runDetect import loadingModel
from utils.general import increment_path

count = 0

detecter = None;


class WindowMain(QMainWindow):

    def __init__(self):
        QMainWindow.__init__(self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.img = []
        self.currentIndex = 0
        self.save_dir = None
        self.img_info = None

        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)

        # 3 top-right button
        self.ui.Bclose.clicked.connect(self.closeFunction)
        self.ui.Bmin.clicked.connect(self.minimizeFunction)

        # open browse button
        self.ui.Bopen.clicked.connect(self.openFunction)

        self.ui.Bdetect.clicked.connect(self.detectFunction)

        self.ui.Bleftt.clicked.connect(self.leftFunctions)
        self.ui.Bleftt.setEnabled(False)
        self.ui.Bright.clicked.connect(self.rightFunctions)
        self.ui.Bright.setEnabled(False)

        self.show()

    def leftFunctions(self):
        if self.currentIndex == 0:
            self.currentIndex = len(self.img) - 1
        else:
            self.currentIndex -= 1

        img_dir = str(self.save_dir / self.img[self.currentIndex])
        self.ui.Limage.setPixmap(QPixmap(img_dir))
        self.ui.Limage.setScaledContents(True)

        self.ui.label_5.setText("Image Info: " + self.img_info[self.currentIndex])

    def rightFunctions(self):
        print("right")
        if self.currentIndex + 1 == len(self.img):
            self.currentIndex = 0
        else:
            self.currentIndex += 1

        img_dir = str(self.save_dir / self.img[self.currentIndex])
        self.ui.Limage.setPixmap(QPixmap(img_dir))
        self.ui.Limage.setScaledContents(True)

        self.ui.label_5.setText("Image Info: " + self.img_info[self.currentIndex])

    def closeFunction(self):
        self.close()

    def minimizeFunction(self):
        self.showMinimized()

    def openFunction(self):
        if not self.ui.checkBox.isChecked():

            fname = QFileDialog.getOpenFileName(self, 'Open file', '',
                                                 "File png (*.png);;File jpg (*.jpg);;All file (*.*)")
            self.ui.lineEdit.setText(fname[0])
            return
        fname = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        self.ui.lineEdit.setText(fname)

    def detectFunction(self):
        global detecter

        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.submit(detecter.startDetect(self.ui.lineEdit.text()))

        print(detecter.save_dir)
        self.save_dir = detecter.save_dir

        self.img = next(walk(self.save_dir), (None, None, []))[2]  # [] if no file
        self.img_info = detecter.info_str
        print(self.img)


        self.ui.label_5.setText("Image Info: "+ self.img_info[-1])

        img_dir = str(self.save_dir / self.img[-1])

        str_img = "image.jpg"

        # loading image
        self.x = QPixmap(img_dir)

        # adding image to label
        self.ui.Limage.setPixmap(self.x)
        self.ui.Limage.setScaledContents(True)

        self.ui.Bleftt.setEnabled(True)
        self.ui.Bright.setEnabled(True)


class WindowLoading(QMainWindow):

    def __init__(self):
        global detecter

        QMainWindow.__init__(self)
        self.ui = None
        self.sth = Ui_LoadingWindow()
        self.sth.setupUi(self)

        detecter = loadingModel()

        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.Progress)
        self.timer.start(50)

        self.show()

    def Progress(self):
        global count

        self.sth.progressBar.setValue(count)

        if count > 100:
            self.timer.stop()
            self.sth = WindowMain()
            self.sth.show()

            self.close()

        count += 1


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = WindowLoading()

    sys.exit(app.exec_())
