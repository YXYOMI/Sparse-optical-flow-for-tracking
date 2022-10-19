'''
run this file to start the program.

Before running, please drag the test video to the video folder and change the PATH in line 11
'''

from PyQt5 import QtGui, QtCore
from PyQt5.QtGui import QPalette, QFont
from main import*
from PyQt5.QtWidgets import *

PATH = "./video/test.mp4"

def on_bt1_clicked():
    w = QWidget()
    screen = QDesktopWidget().screenGeometry()
    size = w.geometry()
    w.move((int)((screen.width() - size.width()) / 2), (int)((screen.height() - size.height()) / 2))
    alert = QMessageBox.information(w,'Alert', 'You chose MODE 1,\n\nBefore you start, please be sure you have read the illustration.', QMessageBox.Ok, QMessageBox.Cancel)
    # alert.setText('You choose MODE 1,\n\nBefore you start, please be sure you have read the illustration.')
    # alert.exec_()
    if alert == QMessageBox.Ok:
        run_mode_1(PATH)
    else:
        return

def on_bt2_clicked():
    # alert = QMessageBox.question()
    # alert.setText('You choose MODE 2,\n\nBefore you start, please be sure you have read the illustration.')
    # alert.exec_()
    # PATH = "./video/test_7.mp4"
    # run_2(PATH)
    w = QWidget()
    screen = QDesktopWidget().screenGeometry()
    size = w.geometry()
    w.move((int)((screen.width() - size.width()) / 2), (int)((screen.height() - size.height()) / 2))
    alert = QMessageBox.information(w, 'Alert','You chose MODE 2,\n\nBefore you start, please be sure you have read the illustration.',QMessageBox.Ok, QMessageBox.Cancel)
    # alert.setText('You choose MODE 1,\n\nBefore you start, please be sure you have read the illustration.')
    # alert.exec_()
    if alert == QMessageBox.Ok:
        run_mode_2(PATH)
    else:
        return

def on_bt3_clicked():
    alert = QMessageBox()
    # label = QLabel("CV Coursework: SPARSE OPTICAL FLOW")
    # label.setFont(QFont('Times', 18, QFont.Bold))
    # alert.addWidget(label_1, alignment=QtCore.Qt.AlignCenter)
    alert.setText('Mode 1:\n\nThis mode does not contain any user interactions.\n\nThe output will contain the green tracking line for moving objects and red rectangle for the contour of the moving objects.\n\n\nMode 2:\n\nWhen running this mode, you should select a rectangle area for tracking (press the left mouse button to select, and press space key to start tracking.)\n\nDuring the video, you can use left mouse button to choose track point.\n\nWARNING: Pleas DO NOT loosen the mouse until you have press space key.\n\n\nIn both mode 1 & 2, you can press \'q\' to stop the program.')
    alert.setFont(QFont('Times', 14, QFont.Light))
    alert.exec_()

def on_bt4_clicked():
    exit()


if __name__ == '__main__':
    app = QApplication([])
    app.setStyle('Windows')

    # app.setStyleSheet("QPushButton { margin: 10ex; }")
    window = QWidget()
    window.setFixedSize(400,200)

    palette = QPalette()
    # palette.setColor(palette.Background,QtGui.QColor(0,0,0))

    bt_1 = QPushButton('MODE 1')
    bt_2 = QPushButton('MODE 2')
    bt_3 = QPushButton('Illustration')
    bt_4 = QPushButton('Exit')

    bt_1.setStyleSheet("QPushButton{color:black}"
                          "QPushButton:hover{color:gray}"
                          # "QPushButton{background-color:rgb(78,255,255)}"
                          # "QPushButton{border:4px}"
                          "QPushButton{padding:2px 4px}")
    bt_1.setMinimumHeight(30)
    bt_1.setMinimumWidth(100)

    bt_2.setStyleSheet("QPushButton{color:black}"
                       "QPushButton:hover{color:gray}"
                       # "QPushButton{background-color:rgb(78,255,255)}"
                       # "QPushButton{border:4px}"
                       "QPushButton{padding:2px 4px}")
    bt_2.setMinimumHeight(30)
    bt_2.setMinimumWidth(100)

    bt_3.setStyleSheet("QPushButton{color:black}"
                       "QPushButton:hover{color:gray}"
                       # "QPushButton{background-color:rgb(78,255,255)}"
                       # "QPushButton{border:4px}"
                       "QPushButton{padding:2px 4px}")
    bt_3.setMinimumWidth(100)

    bt_4.setStyleSheet("QPushButton{color:black}"
                       "QPushButton:hover{color:gray}"
                       # "QPushButton{background-color:rgb(78,255,255)}"
                       # "QPushButton{border:4px}"
                       "QPushButton{padding:2px 4px}")
    bt_4.setMinimumWidth(100)

    layout = QVBoxLayout()
    layout_1 = QVBoxLayout()

    label_1 = QLabel("CV Coursework: SPARSE OPTICAL FLOW")
    label_1.setFont(QFont('Times', 18, QFont.Bold))
    layout_1.addWidget(label_1, alignment=QtCore.Qt.AlignCenter)

    # label_2 = QLabel("20030865")
    # label_1.setFont(QFont('Times', 14))
    # layout_1.addWidget(label_2, alignment=QtCore.Qt.AlignCenter)


    layout_1.addWidget(bt_1,alignment=QtCore.Qt.AlignCenter)
    layout_1.addWidget(bt_2,alignment=QtCore.Qt.AlignCenter)
    # layout_1.setContentsMargins(100,20,100,20)
    layout.addLayout(layout_1)

    layout_2 = QHBoxLayout()

    layout_2.addWidget(bt_3, alignment=QtCore.Qt.AlignLeft)
    layout_2.addWidget(bt_4, alignment=QtCore.Qt.AlignRight)
    layout.addLayout(layout_2)


    # layout_2.setContentsMargins(100, 20, 100, 20)

    # PATH = "./video/test_7.mp4"
    # bt_1.clicked.connect(on_bt1_clicked(PATH))
    # bt_2.clicked.connect(on_bt2_clicked(PATH))
    bt_1.clicked.connect(on_bt1_clicked)
    bt_2.clicked.connect(on_bt2_clicked)
    bt_3.clicked.connect(on_bt3_clicked)
    bt_4.clicked.connect(on_bt4_clicked)

    window.setLayout(layout)
    # window.setLayout(layout_2)
    window.setPalette(palette)
    window.show()
    app.exec_()