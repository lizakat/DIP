import cv2
import os
from img_proc import process_image, rotate_and_place_numbers




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
      


dataset_path = "dataset"

for filename in os.listdir(dataset_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            original_image_path = os.path.join(dataset_path, filename)
            original_image = cv2.imread(original_image_path)
            clear_image = process_image(original_image)
            rotated_result = rotate_and_place_numbers(clear_image)
            #show_img(rotated_result)
            cv2.imwrite(f"dataset_lab3out/{filename}", rotated_result)
            