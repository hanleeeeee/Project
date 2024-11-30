import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.interpolate import RectBivariateSpline

# File paths
path1 = "./input/visible"  # vis files
path2 = "./input/near-infrared"  # nir files
output_path = "./output"
os.makedirs(output_path, exist_ok=True)
files1 = os.listdir(path1)
files2 = os.listdir(path2)

alpha = 1.009
beta = 0.705
count = 0



def global_gamma(img):
    img = np.divide((img - np.min(img)), (np.max(img) - np.min(img)))
    sigma = 0.4
    img = np.power(img, sigma)
    result = (img * 255)
    return result.astype(np.uint8)


for file1 in files1:
    if file1.endswith(".jpg") or file1.endswith(".png"):
        if file1 in files2:
            img1 = cv2.imread(os.path.join(path1, file1))
            img2 = cv2.imread(os.path.join(path2, file1))

            lab_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB).astype(np.float32)
            lab_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB).astype(np.float32)

            L1, a1, b1 = cv2.split(lab_img1)
            L2, a2, b2 = cv2.split(lab_img2)

            a1 = a1.astype(np.float32)
            b1 = b1.astype(np.float32)
            a2 = a2.astype(np.float32)
            b2 = b2.astype(np.float32)
            L1 = L1.astype(np.float32)
            L2 = L2.astype(np.float32)
            temp1 = L1

            diff_L = L2 - L1

            diff_L = np.divide((diff_L - np.min(diff_L)), (np.max(diff_L) - np.min(diff_L)))
            temp = L2 * (1 - diff_L) + L1 * diff_L

            # L3=L1+diff_L
            L3 = np.divide((temp - np.min(temp)), (np.max(temp) - np.min(temp)))
            ##1차 blend완료
            L_blend = L3 * 255

            ##bilateral 해보자
            L3_bilateral = cv2.bilateralFilter(diff_L, -1, 10, 10)

            ##2차 blend완료
            L_blend2 = L_blend * L3_bilateral + (L1) * (1 - L3_bilateral)

            ##모듈clahe hedlq
            L_clahe_global = global_gamma(L_blend2)

            clahe_custom = cv2.createCLAHE(3.0, (8, 8))
            ##형식변환
            L_clahe=L_clahe_global.astype(np.uint8)
            # L_clahe=L_blend2.astype(np.uint8)

            ##clahe삽입 해줌
            clahe_custom_apply = clahe_custom.apply(L_clahe)  # custom 방법

            denominator = (L1 + L2) * 0.5
            gain = np.divide(clahe_custom_apply, denominator)
            CC_gain = alpha * np.power(gain, beta)

            # a, b 채널 처리
            a1 = a1.astype(np.float32) / 128 - 1  # -1 ~ 1 범위로 변환
            b1 = b1.astype(np.float32) / 128 - 1

            a_compensated = a1 * CC_gain
            b_compensated = b1 * CC_gain

            # 정규화
            a_compensated = (a_compensated + 1) / 2 * 255
            b_compensated = (b_compensated + 1) / 2 * 255

            a_compensated = a_compensated.astype(np.uint8)
            b_compensated = b_compensated.astype(np.uint8)
            clahe_custom_apply = clahe_custom_apply.astype(np.uint8)

            # # 채널 합병
            img_clahe = cv2.merge((clahe_custom_apply, a_compensated, b_compensated))  # CLAHE 적용된 L 채널과 보정된 a, b 채널 합병

            img_clahe = cv2.cvtColor(img_clahe, cv2.COLOR_LAB2BGR)

            cv2.imwrite(os.path.join(output_path, file1), img_clahe)
            count = count + 1
            print(f"Processed and saved: {output_path} and the count is: {count}")