# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'loading screen.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_LoadingWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(700, 400)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setContentsMargins(10, 10, 10, 10)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.bg_black = QtWidgets.QFrame(self.centralwidget)
        self.bg_black.setStyleSheet("QFrame{\n"
"    \n"
"    background-color: rgb(47, 52, 55);\n"
"    color : rgb(220,220,220);\n"
"    border-radius: 10px;\n"
"\n"
"}")
        self.bg_black.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.bg_black.setFrameShadow(QtWidgets.QFrame.Raised)
        self.bg_black.setObjectName("bg_black")
        self.Title = QtWidgets.QLabel(self.bg_black)
        self.Title.setGeometry(QtCore.QRect(10, 50, 661, 101))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(40)
        self.Title.setFont(font)
        self.Title.setStyleSheet("color: rgb(244, 119, 255);")
        self.Title.setAlignment(QtCore.Qt.AlignCenter)
        self.Title.setObjectName("Title")
        self.Description = QtWidgets.QLabel(self.bg_black)
        self.Description.setGeometry(QtCore.QRect(10, 150, 661, 41))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(14)
        self.Description.setFont(font)
        self.Description.setStyleSheet("QLabel{\n"
"    color: rgb(222, 251, 255);\n"
"}")
        self.Description.setAlignment(QtCore.Qt.AlignCenter)
        self.Description.setObjectName("Description")
        self.progressBar = QtWidgets.QProgressBar(self.bg_black)
        self.progressBar.setGeometry(QtCore.QRect(150, 250, 391, 23))
        self.progressBar.setStyleSheet("QProgressBar{\n"
"    \n"
"    \n"
"    background-color: rgb(170, 142, 255);\n"
"    color: rgb(200, 200, 200);\n"
"    boder-style:none;\n"
"    border-radius:5px;\n"
"    text-align:center;\n"
"}\n"
"\n"
"\n"
"QProgressBar::chunk{\n"
"    border-radius:5px;\n"
"    background-color: qlineargradient(spread:pad, x1:0.0248756, y1:0.369, x2:1, y2:0.346591, stop:0 rgba(160, 94, 200, 255), stop:1 rgba(94, 220, 255, 255));\n"
"}\n"
"\n"
"\n"
"\n"
"")
        self.progressBar.setProperty("value", 24)
        self.progressBar.setObjectName("progressBar")
        self.loading = QtWidgets.QLabel(self.bg_black)
        self.loading.setGeometry(QtCore.QRect(10, 270, 661, 41))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(12)
        self.loading.setFont(font)
        self.loading.setStyleSheet("QLabel{\n"
"    color: rgb(85, 85, 127);\n"
"    color: rgb(222, 251, 255);\n"
"}")
        self.loading.setAlignment(QtCore.Qt.AlignCenter)
        self.loading.setObjectName("loading")
        self.verticalLayout.addWidget(self.bg_black)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.Title.setText(_translate("MainWindow", "<Strong>Your</Strong> power"))
        self.Description.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-weight:600; font-style:italic;\">Your</span><span style=\" font-style:italic;\"> Description</span></p></body></html>"))
        self.loading.setText(_translate("MainWindow", "loading . . ."))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_LoadingWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
