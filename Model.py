# Load the Dataset file
# Import Packages
# train.csv - datafile contains details image details - id,URL and landmarkid
# Top 10 sampled landmark details are extracted for analysis
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import time
from skimage import io
import os
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
# import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score
import warnings
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication
from LR import Ui_LR
from KNN import Ui_KNN
from NB import Ui_NB
from DT import Ui_DT
from EM import Ui_EM
from SVML import Ui_SVML
from SVMNL import Ui_SVMNL
from RF import Ui_RF
from CVS import Ui_CVS
from LIVE import App

class Ui_Model(object):
    def setupUi(self, Model):
        Model.setObjectName("Model")
        Model.resize(1000, 884)
        self.groupBox = QtWidgets.QGroupBox(Model)
        self.groupBox.setGeometry(QtCore.QRect(10, 20, 1000, 301))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(11)
        self.groupBox.setFont(font)
        self.groupBox.setAlignment(QtCore.Qt.AlignCenter)
        self.groupBox.setObjectName("groupBox")
        self.pushButton = QtWidgets.QPushButton(self.groupBox)
        self.pushButton.setGeometry(QtCore.QRect(10, 50, 240, 51))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.log_reg)
        self.pushButton_8 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_8.setGeometry(QtCore.QRect(270, 50, 211, 51))
        self.pushButton_8.setObjectName("pushButton_8")
        self.pushButton_8.clicked.connect(self.nb)
        self.pushButton_9 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_9.setGeometry(QtCore.QRect(520, 50, 191, 51))
        self.pushButton_9.setObjectName("pushButton_9")
        self.pushButton_9.clicked.connect(self.dt)
        self.pushButton_4 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_4.setGeometry(QtCore.QRect(750, 50, 201, 51))
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_4.clicked.connect(self.rf)
        self.pushButton_3 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_3.setGeometry(QtCore.QRect(10, 120, 201, 51))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_3.clicked.connect(self.svml)
        self.pushButton_5 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_5.setGeometry(QtCore.QRect(270, 120, 240, 51))
        self.pushButton_5.setObjectName("pushButton_5")
        self.pushButton_5.clicked.connect(self.svmnl)
        self.pushButton_2 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_2.setGeometry(QtCore.QRect(520, 120, 210, 51))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.clicked.connect(self.em)
        self.pushButton_6 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_6.setGeometry(QtCore.QRect(750, 120, 201, 51))
        self.pushButton_6.setObjectName("pushButton_6")
        self.pushButton_6.clicked.connect(self.knn)
        self.label = QtWidgets.QLabel(self.groupBox)
        self.label.setGeometry(QtCore.QRect(540, 170, 121, 31))
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.pushButton_7 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_7.setGeometry(QtCore.QRect(475, 220, 280, 71))
        self.pushButton_7.setObjectName("pushButton_7")
        self.pushButton_7.clicked.connect(self.cvs)
        self.pushButton_10 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_10.setGeometry(QtCore.QRect(210, 220, 250, 71))
        self.pushButton_10.setObjectName("pushButton_10")
        self.pushButton_10.clicked.connect(self.live)

        self.retranslateUi(Model)
        QtCore.QMetaObject.connectSlotsByName(Model)

    def retranslateUi(self, Model):
        _translate = QtCore.QCoreApplication.translate
        Model.setWindowTitle(_translate("Model", "Landmark Recognition"))
        self.groupBox.setTitle(_translate("Model", "Select Model"))
        self.pushButton.setText(_translate("Model", "Logistic Regression"))
        self.pushButton_8.setText(_translate("Model", "Naive Bayes"))
        self.pushButton_9.setText(_translate("Model", "Decision Tree"))
        self.pushButton_4.setText(_translate("Model", "Random Forest"))
        self.pushButton_3.setText(_translate("Model", "SVM - Linear"))
        self.pushButton_5.setText(_translate("Model", "SVM - Non-Linear"))
        self.pushButton_2.setText(_translate("Model", "Ensemble Model"))
        self.pushButton_6.setText(_translate("Model", "KNN"))
        self.label.setText(_translate("Model", "(Hard Voting)"))
        self.pushButton_7.setText(_translate("Model", "Cross-Validation Score"))
        self.pushButton_10.setText(_translate("Model", "Live Data Prediction "))

    def log_reg(self):
        Dialog = QtWidgets.QDialog()
        ui = Ui_LR()
        ui.setupUi(Dialog)
        Dialog.show()
        Dialog.exec_()

    def knn(self):
        Dialog = QtWidgets.QDialog()
        ui = Ui_KNN()
        ui.setupUi(Dialog)
        Dialog.show()
        Dialog.exec_()

    def dt(self):
        Dialog = QtWidgets.QDialog()
        ui = Ui_DT()
        ui.setupUi(Dialog)
        Dialog.show()
        Dialog.exec_()

    def rf(self):
        Dialog = QtWidgets.QDialog()
        ui = Ui_RF()
        ui.setupUi(Dialog)
        Dialog.show()
        Dialog.exec_()

    def em(self):
        Dialog = QtWidgets.QDialog()
        ui = Ui_EM()
        ui.setupUi(Dialog)
        Dialog.show()
        Dialog.exec_()

    def svml(self):
        Dialog = QtWidgets.QDialog()
        ui = Ui_SVML()
        ui.setupUi(Dialog)
        Dialog.show()
        Dialog.exec_()

    def svmnl(self):
        Dialog = QtWidgets.QDialog()
        ui = Ui_SVMNL()
        ui.setupUi(Dialog)
        Dialog.show()
        Dialog.exec_()

    def nb(self):
        Dialog = QtWidgets.QDialog()
        ui = Ui_NB()
        ui.setupUi(Dialog)
        Dialog.show()
        Dialog.exec_()

    def cvs(self):
        Dialog = QtWidgets.QDialog()
        ui = Ui_CVS()
        ui.setupUi(Dialog)
        Dialog.show()
        Dialog.exec_()

    def live(self):
        import sys
        app = QApplication(sys.argv)
        liv = App()
        app.show()
        sys.exit(app.exec_())


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Model()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())