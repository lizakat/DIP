import os
import cv2
import numpy as np
from sklearn.cluster import KMeans

def show_img(image, scale_percent=30):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    resized_image = cv2.resize(image, (width, height))
    cv2.imshow("Image", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_image(image, default_area_threshold):
    padding = 10
    padded_image = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    gray = cv2.cvtColor(padded_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 3), 0)
    blurred = cv2.convertScaleAbs(blurred, alpha=1.5, beta=20)

    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 61, 8)

    kernel_dilate = np.ones((7, 7), np.uint8)
    dilated1 = cv2.dilate(binary, kernel_dilate, iterations=1)

    contours, _ = cv2.findContours(dilated1, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    output = np.zeros_like(padded_image)
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if 130000 < contour_area < default_area_threshold:
            cv2.drawContours(output, [contour], -1, (255, 255, 255), 9)
    output = output[padding:-padding, padding:-padding]

    kernel = np.ones((11, 11), np.uint8)
    output = cv2.morphologyEx(output, cv2.MORPH_CLOSE, kernel, iterations=3)

    kernel_dilate = np.ones((9, 9), np.uint8)
    dilated2 = cv2.dilate(output, kernel_dilate, iterations=3)

    kernel_dilate = np.ones((7, 7), np.uint8)
    eroded = cv2.erode(dilated2, kernel_dilate, iterations=4)
    
    gray_image = cv2.cvtColor(eroded, cv2.COLOR_BGR2GRAY)
    
    contours, hierarchy = cv2.findContours(gray_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    internal_contours = np.zeros_like(image)
    #all_contours = np.zeros_like(image)
    result_contours = []

    for i, contour in enumerate(contours):
        #cv2.drawContours(all_contours, [contour], -1, 255, 3)
        if hierarchy[0][i][3] != -1:
            result_contours.append(contour)
            cv2.drawContours(internal_contours, [contour], -1, 255, 3)
    #cv2.imwrite("img1.jpg", all_contours)
    return result_contours

def calc_perimeter(contour):
    perimeter = 0.0
    for i in range(len(contour)):
        x1, y1 = contour[i][0]
        x2, y2 = contour[(i + 1) % len(contour)][0]
        perimeter += np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return perimeter

def calc_area(contour):
    area = 0.0
    for i in range(len(contour)):
        x1, y1 = contour[i][0]
        x2, y2 = contour[(i + 1) % len(contour)][0]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0

def calc_roundness(perimeter, area):
    if perimeter == 0:
        return 0
    return (4 * np.pi * area) / (perimeter ** 2)

def point_line_distance(point, start, end):
    point_3d = np.append(point, 0)
    start_3d = np.append(start, 0)
    end_3d = np.append(end, 0)
    
    if np.array_equal(start, end):
        return np.linalg.norm(point - start)
    else:
        cross_product = np.cross(end_3d - start_3d, start_3d - point_3d)
        distance = np.abs(np.linalg.norm(cross_product) / np.linalg.norm(end_3d - start_3d))
        return distance

def douglas_peucker(contour, epsilon):
    if len(contour) < 3:
        return contour

    start, end = contour[0, 0], contour[-1, 0]
    max_distance = 0
    index = -1
    for i in range(1, len(contour) - 1):
        distance = point_line_distance(contour[i][0], start, end)
        if distance > max_distance:
            index = i
            max_distance = distance

    if max_distance > epsilon:
        left = douglas_peucker(contour[:index+1], epsilon)
        right = douglas_peucker(contour[index:], epsilon)
        return np.vstack((left[:-1], right))
    else:
        return np.array([contour[0], contour[-1]])

def find_polygon_corners(contour, perimeter, epsilon_factor=0.04):
    epsilon = epsilon_factor * perimeter
    approx = douglas_peucker(contour, epsilon)
    return len(approx) - 1

def calc_properties_manual(contour):
    perimeter = calc_perimeter(contour)
    area = calc_area(contour)
    corners = find_polygon_corners(contour, perimeter)
    roundness = calc_roundness(perimeter, area)
    return [roundness, corners], perimeter

def compare_contour_features(contour):
    perimeter_cv2 = cv2.arcLength(contour, True)
    area_cv2 = cv2.contourArea(contour)
    approx = cv2.approxPolyDP(contour, 0.04 * perimeter_cv2, True)
    corners_cv2 = len(approx)
    roundness_cv2 = (4 * np.pi * area_cv2) / (perimeter_cv2 ** 2) if perimeter_cv2 != 0 else 0

    perimeter_custom = calc_perimeter(contour)
    area_custom = calc_area(contour)
    corners_custom = find_polygon_corners(contour, perimeter_custom)
    roundness_custom = calc_roundness(perimeter_custom, area_custom)

    print("Сравнение характеристик контура:")
    print(f"Периметр (OpenCV): {perimeter_cv2:.2f}, (Собственный): {perimeter_custom:.2f}")
    print(f"Площадь (OpenCV): {area_cv2:.2f}, (Собственная): {area_custom:.2f}")
    print(f"Количество углов (OpenCV): {corners_cv2}, (Собственное): {corners_custom}")
    print(f"Округлость (OpenCV): {roundness_cv2:.2f}, (Собственная): {roundness_custom:.2f}")

def train_kmeans(n_clusters=3):
    training_data = [
        [0.83, 4], [0.94, 8], [0.76, 6], [0.82, 4],
        [0.95, 8], [0.78, 6], [0.75, 4], [0.80, 6],
        [0.71, 4], [0.63, 4], [0.73, 4]
    ]
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(training_data)
    return kmeans

def classify_contours(contours, kmeans, image):
    output_image = image.copy()
    for contour in contours:
        props, perimeter = calc_properties_manual(contour)
        #compare_contour_features(contour)

        predicted_label = kmeans.predict([props])[0]

        if predicted_label == 0:
            new_hue = 0
        elif predicted_label == 1:
            new_hue = 30
        elif predicted_label == 2:
            new_hue = 60
        else:
            new_hue = 120

        output_image = recolor(output_image, new_hue, contour)
        print(f"Контур: (Углы: {props[1]}, Округлость: {props[0]:.2f}, Периметр: {perimeter:.2f})")
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

kmeans = train_kmeans()

dataset_path = "dataset_lab1out"
output_path = "dataset_lab2out_kmeans"

if not os.path.exists(output_path):
    os.makedirs(output_path)

for filename in os.listdir(dataset_path):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        original_image_path = os.path.join(dataset_path, filename)
        original_image = cv2.imread(original_image_path)

        height, width = original_image.shape[:2]
        current_area = width * height
        scale_factor = (default_area / current_area) ** 0.5
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        resized_image = cv2.resize(original_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        contours = process_image(resized_image, default_area)
        final_image = classify_contours(contours, kmeans, resized_image)
        
        final_image = cv2.resize(final_image, (width, height), interpolation=cv2.INTER_LINEAR)
        
        show_img(final_image)
        print("----------------------------------------------------------")
        cv2.imwrite(os.path.join(output_path, filename), final_image)