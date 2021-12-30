# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'process_green.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setStyleSheet("QFrame{\n"
"    border-radius:10px;\n"
"    boder-image:url(:/pic/wave.png);\n"
"    \n"
"    background-color: rgb(47, 52, 55);\n"
"\n"
"}")
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.Bclose = QtWidgets.QPushButton(self.frame)
        self.Bclose.setGeometry(QtCore.QRect(720, 0, 61, 31))
        self.Bclose.setStyleSheet("QPushButton{\n"
"    color:white;\n"
"    border-radius: 10px;\n"
"    background-color: rgb(47, 52, 55);\n"
"    border:none;    \n"
"    outline:none;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"    background-color: #64b5f6;\n"
"}")
        self.Bclose.setObjectName("Bclose")
        self.Bmax = QtWidgets.QPushButton(self.frame)
        self.Bmax.setGeometry(QtCore.QRect(660, 0, 61, 31))
        self.Bmax.setStyleSheet("QPushButton{\n"
"    color:white;\n"
"    border-radius:10px;\n"
"    background-color: rgb(47, 52, 55);\n"
"    border:none;    \n"
"    outline:none;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"    background-color: #64b5f6;\n"
"}")
        self.Bmax.setObjectName("Bmax")
        self.Bmin = QtWidgets.QPushButton(self.frame)
        self.Bmin.setGeometry(QtCore.QRect(600, 0, 61, 31))
        self.Bmin.setStyleSheet("QPushButton{\n"
"    color:white;\n"
"    border-radius:10px;\n"
"    background-color: rgb(47, 52, 55);\n"
"    border:none;    \n"
"    outline:none;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"    background-color: #64b5f6;\n"
"}")
        self.Bmin.setObjectName("Bmin")
        self.Bopen = QtWidgets.QPushButton(self.frame)
        self.Bopen.setGeometry(QtCore.QRect(550, 110, 101, 31))
        self.Bopen.setStyleSheet("QPushButton{\n"
"    font-size: 17px;\n"
"    color:white;\n"
"    background-color: #6E3CBC;\n"
"    border-radius:10px;\n"
"}\n"
"\n"
"QPushButton:hover{\n"
"    background:cyan;\n"
"    color:black;\n"
"}")
        self.Bopen.setObjectName("Bopen")
        self.lineEdit = QtWidgets.QLineEdit(self.frame)
        self.lineEdit.setGeometry(QtCore.QRect(120, 110, 401, 31))
        self.lineEdit.setStyleSheet("QLineEdit{\n"
"    border-radius:10px;\n"
"}")
        self.lineEdit.setObjectName("lineEdit")
        self.label_2 = QtWidgets.QLabel(self.frame)
        self.label_2.setGeometry(QtCore.QRect(90, 150, 601, 20))
        self.label_2.setStyleSheet("QLabel{\n"
"    border:none;\n"
"    border-bottom: 1px solid grey;\n"
"\n"
"}")
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.frame)
        self.label_3.setGeometry(QtCore.QRect(120, 60, 141, 41))
        font = QtGui.QFont()
        font.setFamily("MV Boli")
        font.setPointSize(15)
        self.label_3.setFont(font)
        self.label_3.setStyleSheet("QLabel{\n"
"    color: #F5F5F5;\n"
"}")
        self.label_3.setObjectName("label_3")
        self.frame_2 = QtWidgets.QFrame(self.frame)
        self.frame_2.setGeometry(QtCore.QRect(120, 200, 611, 331))
        self.frame_2.setStyleSheet("QFrame{\n"
"    background-color:#1A374D;\n"
"    border: 2px solid  white;\n"
"    box-shadow: 12px 12px 2px 1px rgba(0, 0, 255, .2);\n"
"}")
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.Bdetect = QtWidgets.QPushButton(self.frame_2)
        self.Bdetect.setGeometry(QtCore.QRect(300, 40, 181, 41))
        self.Bdetect.setStyleSheet("QPushButton{\n"
"    font-size: 17px;\n"
"    color:white;\n"
"    background-color: #6E3CBC;\n"
"    border-radius:10px;\n"
"}\n"
"\n"
"QPushButton:hover{\n"
"    background:cyan;\n"
"    color:black;\n"
"}")
        self.Bdetect.setObjectName("Bdetect")
        self.Limage = QtWidgets.QLabel(self.frame_2)
        self.Limage.setGeometry(QtCore.QRect(60, 30, 161, 181))
        self.Limage.setStyleSheet("QLabel{\n"
"    \n"
# "    border: 1px solid grey;\n"
  "border-image: url(image.jpg);"
"}\n"
"")
        self.Limage.setText("")
        self.Limage.setObjectName("Limage")
        self.label_5 = QtWidgets.QLabel(self.frame_2)
        self.label_5.setGeometry(QtCore.QRect(50, 240, 491, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_5.setFont(font)
        self.label_5.setStyleSheet("QLabel{\n"
"    Color: #F5F5F5;\n"
"}\n"
"")
        self.label_5.setObjectName("label_5")
        self.Bview = QtWidgets.QPushButton(self.frame_2)
        self.Bview.setGeometry(QtCore.QRect(300, 100, 131, 41))
        self.Bview.setStyleSheet("QPushButton{\n"
"    font-size: 17px;\n"
"    color:white;\n"
"    background-color: #6E3CBC;\n"
"    border-radius:10px;\n"
"}\n"
"\n"
"QPushButton:hover{\n"
"    background:cyan;\n"
"    color:black;\n"
"}")
        self.Bview.setObjectName("Bview")
        self.Bright = QtWidgets.QPushButton(self.frame_2)
        self.Bright.setGeometry(QtCore.QRect(230, 110, 41, 41))
        self.Bright.setStyleSheet("QPushButton{\n"
"    Background-color: #7CD1B8;\n"
"    border-radius:20px;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"    background-color: #64b5f6;\n"
"}")
        self.Bright.setObjectName("Bright")
        self.Bleftt = QtWidgets.QPushButton(self.frame_2)
        self.Bleftt.setGeometry(QtCore.QRect(10, 110, 41, 41))
        self.Bleftt.setStyleSheet("QPushButton{\n"
"    Background-color: #7CD1B8;\n"
"    border-radius:20px;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"    background-color: #64b5f6;\n"
"}")
        self.Bleftt.setObjectName("Bleftt")
        self.checkBox = QtWidgets.QCheckBox(self.frame)
        self.checkBox.setGeometry(QtCore.QRect(660, 120, 81, 20))
        self.checkBox.setStyleSheet("QCheckBox{\n"
"    font-size:15px;\n"
"    color: grey;\n"
"}")
        self.checkBox.setObjectName("checkBox")
        self.verticalLayout.addWidget(self.frame)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.Bclose.setToolTip(_translate("MainWindow", "<html><head/><body><p>❌</p></body></html>"))
        self.Bclose.setText(_translate("MainWindow", "❌"))
        self.Bmax.setText(_translate("MainWindow", "🔲"))
        self.Bmin.setText(_translate("MainWindow", "➖"))
        self.Bopen.setToolTip(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:12pt;\">📂</span></p></body></html>"))
        self.Bopen.setText(_translate("MainWindow", "📂 Browse"))
        self.label_3.setText(_translate("MainWindow", "Choose file:"))
        self.Bdetect.setToolTip(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:12pt;\">🤖</span></p></body></html>"))
        self.Bdetect.setText(_translate("MainWindow", "🤖 Start detect"))
        self.label_5.setText(_translate("MainWindow", "Image Info:"))
        self.Bview.setToolTip(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:12pt;\">👀</span></p></body></html>"))
        self.Bview.setText(_translate("MainWindow", "👀 View"))
        self.Bright.setText(_translate("MainWindow", "▶️"))
        self.Bleftt.setText(_translate("MainWindow", "◀️"))
        self.checkBox.setText(_translate("MainWindow", "Folder"))
import picture_rc
