import cv2
import numpy as np

def process_image(original_image):

    height, width = original_image.shape[:2]

    default_width = 4624
    default_height = 2080
    default_area = default_width * default_height

    current_area = width * height
    scale_factor = current_area / default_area

    min_contour_area = int(80000 * scale_factor)

    hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([100, 180, 0])
    upper_blue = np.array([140, 250, 255])

    lower_salad = np.array([70, 170, 0])
    upper_salad = np.array([110, 255, 255])

    blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
    salad_mask = cv2.inRange(hsv_image, lower_salad, upper_salad)

    combined_mask = cv2.bitwise_or(blue_mask, salad_mask)

    kernel = np.ones((5, 5), np.uint8)
    combined_mask_cleaned = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(combined_mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

    filtered_mask = np.zeros_like(combined_mask_cleaned)

    cv2.drawContours(filtered_mask, filtered_contours, -1, (255), thickness=cv2.FILLED)

    isolated_figures = cv2.bitwise_and(original_image, original_image, mask=filtered_mask)

    sobelx = cv2.Sobel(filtered_mask, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(filtered_mask, cv2.CV_64F, 0, 1, ksize=5)
    sobel_edges = cv2.magnitude(sobelx, sobely)

    sobel_edges = cv2.normalize(sobel_edges, None, 0, 255, cv2.NORM_MINMAX)
    sobel_edges = sobel_edges.astype(np.uint8)

    blurred_edges = cv2.GaussianBlur(sobel_edges, (5, 5), 0)

    red_boundary = np.zeros_like(original_image)
    red_boundary[blurred_edges > 50] = [0, 0, 255]  # Red color in BGR

    final_image = cv2.addWeighted(isolated_figures, 1, red_boundary, 0, 0)

    return final_image


if __name__ == "__main__":

    original_image_path = 'dataset/1695138157785.jpg'
    original_image = cv2.imread(original_image_path)

    final_image = process_image(original_image)
    output_path = f'output_image/final_image_1.jpg'  # Путь для сохранения
    cv2.imwrite(output_path, final_image)  # Сохранение оригинального финального изображе

    scale_percent = 20  # Resize scale
    width = int(final_image.shape[1] * scale_percent / 100)
    height = int(final_image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv2.resize(final_image, dim)

    # Show the resized final image
    cv2.imshow("Final Image with Smooth Red Boundaries", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
