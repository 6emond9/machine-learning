"""
OCR_number
"""

import numpy as np
import cv2
import pytesseract
import matplotlib.pyplot as plt


class NumberOCR:
    def __init__(self, file_name):
        self.file_name = file_name

    def processing(self):
        # 读取图片为灰度格式并查看
        img = cv2.imread(self.file_name)
        plt.imshow(img)
        plt.xticks([]), plt.yticks([])
        plt.show()

        # 将图像进行黑白二值处理提高文字的可辨识度
        ret, img2 = cv2.threshold(np.array(img), 128, 255, cv2.THRESH_BINARY)
        plt.imshow(img2)
        plt.xticks([]), plt.yticks([])
        plt.show()
        return img

    def ocr(self, img):
        result_str = pytesseract.image_to_string(img)
        return result_str

    def run(self):
        img = self.processing()
        result_str = self.ocr(img)
        # print(type(result_str), len(result_str))
        print(result_str)
        # print(result_str.split())


if __name__ == '__main__':
    file = './db/img.png'
    ocr = NumberOCR(file)
    ocr.run()
