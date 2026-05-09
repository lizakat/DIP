import cv2
import numpy as np


def show_img(final_image):
    scale_percent = 20  # Resize scale
    width = int(final_image.shape[1] * scale_percent / 100)
    height = int(final_image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv2.resize(final_image, dim)

    # Show the resized final image
    cv2.imshow("Final Image with Smooth Red Boundaries", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
def process_image(original_image):

    hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)

    lower_salad = np.array([40, 140, 130])
    upper_salad = np.array([90, 255, 255])

    mask = cv2.inRange(hsv_image, lower_salad, upper_salad)
    show_img(mask)
    kernel = np.ones((5, 5), np.uint8)
    combined_mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    isolated_figures = cv2.bitwise_and(original_image, original_image, mask=combined_mask_cleaned)
    
    return isolated_figures


def rotate_and_place_numbers(no_background):
    height, width = no_background.shape[:2]

    default_width = 4624
    default_height = 2080
    default_area = default_width * default_height

    current_area = width * height
    scale_factor = current_area / default_area

    min_num_area = int(15000 * scale_factor)
    
    no_background_gray = cv2.cvtColor(no_background, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(no_background_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Фильтрация контуров по площади
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > min_num_area]


    # Создаем пустое изображение для результата
    rotated_result = np.zeros_like(no_background)

    for contour in filtered_contours:
        # Получение минимального ограничивающего прямоугольника
        rect_properties = cv2.minAreaRect(contour)
        rotation_angle = rect_properties[2]
        rect_dimensions = rect_properties[1]

        # Проверка ориентации прямоугольника и корректировка угла
        if rect_dimensions[0] > rect_dimensions[1]:
            rotation_angle += 90
            rect_dimensions = (rect_dimensions[1], rect_dimensions[0])

        # Поворот изображения на рассчитанный угол
        center_point = tuple(map(int, rect_properties[0]))
        transformation_matrix = cv2.getRotationMatrix2D(center_point, rotation_angle, 1.0)
        rotated_section = cv2.warpAffine(
            no_background,
            transformation_matrix,
            (no_background.shape[1], no_background.shape[0])
        )

        # Извлечение области интереса (ROI)
        box_center = tuple(map(int, rect_properties[0]))
        rect_width, rect_height = map(int, rect_dimensions)
        roi_x = box_center[0] - rect_width // 2
        roi_y = box_center[1] - rect_height // 2

        # Определение границ области с учётом границ изображения
        crop_region = (
            max(0, roi_x),
            max(0, roi_y),
            min(rotated_section.shape[1], rect_width),
            min(rotated_section.shape[0], rect_height)
        )

        # Вырезание нужного участка
        extracted_roi = rotated_section[
            crop_region[1]:crop_region[1] + crop_region[3],
            crop_region[0]:crop_region[0] + crop_region[2]
        ]

        # Вставка вырезанного участка на итоговое изображение
        if extracted_roi.size > 0:
            rotated_result[
                crop_region[1]:crop_region[1] + crop_region[3],
                crop_region[0]:crop_region[0] + crop_region[2]
            ] = extracted_roi

    return rotated_result




if __name__ == "__main__":

    original_image_path = 'dataset/11.jpg'
    original_image = cv2.imread(original_image_path)

    clear_image = process_image(original_image)
    # Поворот и размещение цифр
    rotated_result = rotate_and_place_numbers(clear_image)

    output_path = f'dataset_lab3out/11.jpg'  
    cv2.imwrite(output_path, rotated_result)  