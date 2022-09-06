
from scipy import ndimage
from PyQt5 import QtWidgets, QtCore, QtGui, QtMultimedia
from PyQt5 import QtGui
import numpy as np
import pandas as pd
from PyQt5 import QtWidgets as qtw
import pyqtgraph as pg
import matplotlib.pyplot as plt
from images_filtering import Ui_MainWindow
from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import QFileDialog, QLabel
import sys
from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel, QTextEdit
import sys
from PyQt5.QtGui import QPixmap
from matplotlib import pyplot as plt
from math import sqrt
from skimage.color import rgb2gray
from skimage.io import imread
import cv2
import numpy as np
from PyQt5.QtGui import QImage, QPixmap, QPalette, QPainter
from PyQt5.QtWidgets import QLabel, QSizePolicy, QScrollArea, QMessageBox, QMainWindow, QMenu, QAction
from PIL import Image
from scipy import fftpack
import scipy.fftpack as fp
QMediaContent = QtMultimedia.QMediaContent
QMediaPlayer = QtMultimedia.QMediaPlayer
pauseCounter = False


class MainWindow(qtw.QMainWindow):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.actionGray_Scale_4.triggered.connect(
            lambda: self.median_filter())
        self.ui.actionGray_Scale_6.triggered.connect(
            lambda: self.laplacian_filter())

        self.ui.actionHistogram_equlizer.triggered.connect(
            lambda: self.get_histogram())
        self.ui.actionOpen.triggered.connect(lambda: self.open_img())

        self.ui.actionGray_Scale_2.triggered.connect(
            lambda: self.high_filter_frequency())
        self.ui.actionGrary_Scale.triggered.connect(
            lambda: self.low_filter_frequency())

        self.ui.actionGray_Scale_3.triggered.connect(lambda: self.lp_spatial())
        self.ui.actionGray_Scale.triggered.connect(lambda: self.hp_spatial())
        # self.ui.histogram_buttom.clicked.connect(lambda: self.grayScale())

    def open_img(self):
        """browses for any image on local machine"""

        options = QFileDialog.Options()
        # fileName = QFileDialog.getOpenFileName(self, "Open File", QDir.currentPath())
        self.fileName, _ = QFileDialog.getOpenFileName(self, 'QFileDialog.getOpenFileName()', '',
                                                       'Images (*.png *.jpeg *.jpg *.bmp *.gif)', options=options)

        if self.fileName:
            print(self.fileName)
            self.image = QImage(self.fileName)
            if self.image.isNull():
                QMessageBox.information(
                    self, "Image Viewer", "Cannot load %s." % self.fileName)
                return

            self.ui.normal_image.setPixmap(
                QPixmap.fromImage(self.image).scaled(600, 328))

    def convert_to_gray(self):
        """Converts rgb pics to gray scale"""
        self.img = cv2.imread(self.fileName)
        self.grayImg = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        plt.imshow(self.grayImg, 'gray')
        plt.axis('off')
        plt.savefig('gray.jpg', bbox_inches='tight')
        # plt.show()
        pixmap = QPixmap('gray.jpg').scaled(600, 308)
        self.ui.normal_image.setPixmap(pixmap)

    def display_pic(self, ui_element, pic, pic_name):
        """displays a certain pic on gui"""
        plt.imshow(pic, 'gray')
        plt.axis('off')
        plt.savefig('{}.jpg'.format(pic_name), bbox_inches='tight')
        # plt.show()
        pixmap = QPixmap('{}.jpg'.format(pic_name)).scaled(624, 452)
        ui_element.setPixmap(pixmap)
        pass

    def lp_spatial(self):
        """Applies low pss filter in spatial domain"""
        self.convert_to_gray()
        self.kernel = np.ones((20, 20), np.float32)/400
        self.filtered_img = cv2.filter2D(self.grayImg, -1, self.kernel)
        self.display_pic(self.ui.after_filter,
                         self.filtered_img, 'filtered_lp_spatial')

    def hp_spatial(self):
        """Applies high pss filter in spatial domain"""
        self.convert_to_gray()
        self.kernel = np.ones((50, 50), np.float32)/(50*50)
        self.filtered_img = cv2.filter2D(self.grayImg, -1, self.kernel)
        self.filtered_img = self.grayImg.astype(
            'float32') - self.filtered_img.astype('float32')
        self.filtered_img = self.filtered_img + \
            127*np.ones(self.grayImg.shape, np.uint8)
        self.display_pic(self.ui.after_filter,
                         self.filtered_img, 'filtered_hp_spatial')

    def high_filter_frequency(self):
        """Applies hp filter in freq domain"""
        img = np.mean(imread(self.fileName), axis=2)  # assuming an RGB image
        gray = cv2.imdecode(np.fromfile(self.fileName, dtype=np.uint8), 1)
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        self.display_pic(self.ui.normal_image, gray, 'gray')
        # self.convert_to_gray()
        F1 = fftpack.fft2((img).astype(float))
        F2 = fftpack.fftshift(F1)
        (w, h) = img.shape
        half_w, half_h = int(w/2), int(h/2)

        # high pass filter
        n = 5
        # select all but the first 50x50 (low) frequencies
        F2[half_w-n:half_w+n+1, half_h-n:half_h+n+1] = 0
        im1 = fp.ifft2(fftpack.ifftshift(F2)).real
        self.display_pic(self.ui.after_frequency_filter, im1, 'high_freq')

    def low_filter_frequency(self):
        """Applies lp filter in freq domain"""

        img = cv2.imread(self.fileName)[:, :, 0]  # gray-scale image
        gray = cv2.imdecode(np.fromfile(self.fileName, dtype=np.uint8), 1)
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        self.display_pic(self.ui.normal_image, gray, 'gray')

        # self.convert_to_gray()
        (w, h) = img.shape
        img = img[:w, :w]  # crop to 700 x 700
        r = 300  # how narrower the window is
        ham = np.hamming(w)[:, None]  # 1D hamming
        ham2d = np.sqrt(np.dot(ham, ham.T)) ** r  # expand to 2D hamming
        f = cv2.dft(img.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
        f_shifted = np.fft.fftshift(f)
        f_complex = f_shifted[:, :, 0]*1j + f_shifted[:, :, 1]
        f_filtered = ham2d * f_complex

        f_filtered_shifted = np.fft.fftshift(f_filtered)
        inv_img = np.fft.ifft2(f_filtered_shifted)  # inverse F.T.
        filtered_img = np.abs(inv_img)
        filtered_img -= filtered_img.min()
        filtered_img = filtered_img*255 / filtered_img.max()
        filtered_img = filtered_img.astype(np.uint8)
        self.display_pic(self.ui.after_frequency_filter, filtered_img, 'img3')

    def median_filter(self):
        """Applies median filter on pics"""
        img_noisy1 = np.mean(imread(self.fileName), axis=2)
        m, n = img_noisy1.shape
        self.filtered_img = np.zeros([m, n])
        for i in range(1, m-1):
            for j in range(1, n-1):
                temp = [img_noisy1[i-1, j-1],
                        img_noisy1[i-1, j],
                        img_noisy1[i-1, j + 1],
                        img_noisy1[i, j-1],
                        img_noisy1[i, j],
                        img_noisy1[i, j + 1],
                        img_noisy1[i + 1, j-1],
                        img_noisy1[i + 1, j],
                        img_noisy1[i + 1, j + 1]]

                temp = sorted(temp)
                self.filtered_img[i, j] = temp[4]
        self.filtered_img = self.filtered_img.astype(np.uint8)
        self.display_pic(self.ui.after_filter,
                         self.filtered_img, 'filtered_median_spatial')

    def laplacian_filter(self):
        """Applies laplacian filter on pics"""

        img_edges = np.mean(imread(self.fileName), axis=2)
        self.filtered_img = cv2.Laplacian(img_edges, cv2.CV_64F, ksize=3)
        self.filtered_img = cv2.convertScaleAbs(self.filtered_img)
        self.display_pic(self.ui.after_filter, self.filtered_img,
                         'filtered_laplacian_spatial')

    def get_histogram(self):
        """Performs histogram Equalization"""
        self.convert_to_gray()
        # convert to  array
        img_array = np.asarray(self.grayImg)
        histogram_array = np.bincount(img_array.flatten(), minlength=256)

        # normalization
        num_pixels = np.sum(histogram_array)
        histogram_array = histogram_array/num_pixels

        # cumulative histogram
        cumulative_array = np.cumsum(histogram_array)
        transform_map = np.floor(255 * cumulative_array).astype(np.uint8)
        # image array into 1D list
        img_list = list(img_array.flatten())

        # transform pixel values to equalize
        eq_img_list = [transform_map[p] for p in img_list]

        # reshape to img_array
        eq_img_array = np.reshape(np.asarray(eq_img_list), img_array.shape)
        eq_histogram_array = np.bincount(eq_img_array.flatten(), minlength=256)
        num_pixels = np.sum(eq_histogram_array)
        eq_pdf = eq_histogram_array/num_pixels

        # plot
        plt.figure()
        plt.plot(histogram_array)
        plt.plot(eq_pdf)
        plt.xlabel('Pixel intensity')
        plt.ylabel('Distribution')
        plt.legend(['Original', 'Equalized'])
        plt.savefig('Histogram.jpg', bbox_inches='tight')
        # plt.show()
        pixmap = QPixmap('Histogram.jpg').scaled(600, 308)
        self.ui.histogram.setPixmap(pixmap)

        eq_img = Image.fromarray(eq_img_array, mode='L')
        eq_img.save('Equalized.jpg')
        pixmap = QPixmap('Equalized.jpg').scaled(600, 308)
        self.ui.after_histogram.setPixmap(pixmap)


if __name__ == '__main__':
    app = qtw.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())
