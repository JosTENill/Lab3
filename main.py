import cv2
import numpy as np
import os


# Функція зсуву зображення
def shift_image(image, x_shift, y_shift):
    h, w = image.shape[:2]
    M = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    shifted_image = cv2.warpAffine(image, M, (w, h))
    return shifted_image


# Функція інверсії
def invert_image(image):
    return 255 - image


# Функція згладжування по Гауссу
def gaussian_blur(image, kernel_size):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


# Функція розмиття "рух по діагоналі"
def motion_blur(image, kernel_size):
    kernel = np.zeros((kernel_size, kernel_size))
    np.fill_diagonal(kernel, 1)
    kernel /= kernel_size
    return cv2.filter2D(image, -1, kernel)


# Функція покращення різкості
def sharpen_image(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)


# Собелівський фільтр
def sobel_filter(image):
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(grad_x, grad_y)
    return np.uint8(sobel)


# Виділення границь
def edge_filter(image):
    return cv2.Canny(image, 100, 200)


# Користувацький фільтр (приклад: художній ефект)
def custom_filter(image):
    kernel = np.array([[1, 1, 1],
                       [1, -7, 1],
                       [1, 1, 1]])
    return cv2.filter2D(image, -1, kernel)


# Основна функція
def main():
    input_path = 'p.jpg'  # Вкажіть шлях до зображення на робочому столі
    output_dir = 'results'  # Папка для результатів (повинна бути не на робочому столі)
    os.makedirs(output_dir, exist_ok=True)

    # Завантаження зображення
    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)  # Використання градації сірого
    if image is None:
        print("Не вдалося завантажити зображення!")
        return

    # 1. Зсув
    shifted = shift_image(image, 10, 20)
    cv2.imwrite(os.path.join(output_dir, 'shifted.jpg'), shifted)

    # 2. Інверсія
    inverted = invert_image(image)
    cv2.imwrite(os.path.join(output_dir, 'inverted.jpg'), inverted)

    # 3. Згладжування по Гауссу
    gaussian = gaussian_blur(image, 11)
    cv2.imwrite(os.path.join(output_dir, 'gaussian_blur.jpg'), gaussian)

    # 4. Розмиття "рух по діагоналі"
    motion = motion_blur(image, 7)
    cv2.imwrite(os.path.join(output_dir, 'motion_blur.jpg'), motion)

    # 5. Покращення різкості
    sharpened = sharpen_image(image)
    cv2.imwrite(os.path.join(output_dir, 'sharpened.jpg'), sharpened)

    # 6. Собелівський фільтр
    sobel = sobel_filter(image)
    cv2.imwrite(os.path.join(output_dir, 'sobel.jpg'), sobel)

    # 7. Фільтр границь
    edges = edge_filter(image)
    cv2.imwrite(os.path.join(output_dir, 'edges.jpg'), edges)

    # 8. Користувацький фільтр
    custom = custom_filter(image)
    cv2.imwrite(os.path.join(output_dir, 'custom_filter.jpg'), custom)

    print(f"Обробка завершена. Результати збережено в папці '{output_dir}'.")


if __name__ == '__main__':
    main()
