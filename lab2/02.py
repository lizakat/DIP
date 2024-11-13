import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def show_img(image, scale_percent=30):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    resized_image = cv2.resize(image, (width, height))
    cv2.imshow("Image", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_image(image, default_area_threshold, scale_factor):

    padding = 10
    padded_image = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    gray = cv2.cvtColor(padded_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 3), 0)
    blurred = cv2.convertScaleAbs(blurred, alpha=1.5, beta=20)

    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 61, 8)
    show_img(binary)

    kernel_dilate = np.ones((7, 7), np.uint8)
    binary = cv2.dilate(binary, kernel_dilate, iterations=1)
    #show_img(binary)

    contours, h = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    output = np.zeros_like(padded_image)
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if 130000 * scale_factor < contour_area < default_area_threshold:
            cv2.drawContours(output, [contour], -1, (255, 255, 255), 4)
    output = output[padding:-padding, padding:-padding]
    #show_img(output)

    kernel = np.ones((13, 13), np.uint8)
    output = cv2.morphologyEx(output, cv2.MORPH_CLOSE, kernel, iterations=4)
    #show_img(output)
    kernel_dilate = np.ones((9, 9), np.uint8)
    binary = cv2.dilate(output, kernel_dilate, iterations=3)

    kernel_dilate = np.ones((7, 7), np.uint8)
    binary = cv2.erode(binary, kernel_dilate, iterations=4)
    
    binary = cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY)
    #show_img(binary)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    internal_contours = np.zeros_like(image)
    all_contours = np.zeros_like(image)
    result_contours = []

    for i, contour in enumerate(contours):
        cv2.drawContours(all_contours, [contour], -1, 255, 3)
        if hierarchy[0][i][3] != -1:
            result_contours.append(contour)
            cv2.drawContours(internal_contours, [contour], -1, 255, 3)
    #show_img(all_contours)
    #show_img(internal_contours)
    cv2.imwrite("img1.jpg", all_contours)
    return result_contours

def calc_properties(contour):
    perimeter = cv2.arcLength(contour, True)
    corners = len(cv2.approxPolyDP(contour, 0.04 * perimeter, True))
    area = cv2.contourArea(contour)
    roundness = (4 * np.pi * area) / (perimeter ** 2) if perimeter != 0 else 0
    return [roundness, corners]

def train_knn():
    training_data = [
        [0.83, 4], [0.94, 8], [0.76, 6], [0.82, 4],
        [0.95, 8], [0.78, 6], [0.75, 4], [0.80, 6],
        [0.71, 4], [0.63, 4], [0.73, 4]
    ]
    labels = [2, 1, 3, 2, 1, 3, 2, 3, 2, 2, 2]
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(training_data, labels)
    return knn

def classify_contours(contours, knn, image):
    output_image = image.copy()
    for contour in contours:
        props = calc_properties(contour)
        predicted_label = knn.predict([props])[0]

        if predicted_label == 1:
            new_hue, shape_name = 0, "Circle"
        elif predicted_label == 2:
            new_hue, shape_name = 30, "Quadrilateral"
        elif predicted_label == 3:
            new_hue, shape_name = 60, "Hexagon"
        else:
            new_hue, shape_name = 120, "-"

        output_image = recolor(output_image, new_hue, contour)
        print(f"Контур: {shape_name} (Углы: {props[1]}, Округлость: {props[0]:.2f})")
    return output_image


def recolor(image, new_hue, contour):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    kernel_dilate = np.ones((5, 5), np.uint8)
    
    cv2.fillPoly(mask, [contour], (255, 255, 255))
    mask = cv2.dilate(mask, kernel_dilate, iterations=2)
    hsv_image[mask == 255, 0] = new_hue
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

default_width, default_height = 4624, 2080
default_area = default_width * default_height - 15000


knn = train_knn()

original_image_path = "dataset_lab1out/5.jpg"

original_image = cv2.imread(original_image_path)
img_copy = original_image.copy()

height, width = img_copy.shape[:2]
current_area = width * height
scale_factor = current_area / default_area

contours = process_image(original_image, default_area, scale_factor)
final_image = classify_contours(contours, knn, img_copy)

show_img(final_image)