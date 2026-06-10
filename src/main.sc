import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt

# ==============================================================================
# УКАЖИТЕ ПРАВИЛЬНОЕ ИМЯ ФАЙЛА ИЗОБРАЖЕНИЯ
# ==============================================================================
IMAGE_PATH = 'week3_image_1.jpg'  # Замените на актуальное имя файла, если нужно

# 1. Считайте изображение и преобразуйте в палитру RGB
img_bgr = cv2.imread(IMAGE_PATH)
if img_bgr is None:
    raise FileNotFoundError(f"Не удалось найти изображение {IMAGE_PATH}. Проверьте имя файла и путь к нему.")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# 2. Сожмите изображение до ширины 2304, соблюдая пропорции
target_width = 2304
original_height, original_width = img_rgb.shape[:2]

# Коэффициент отношения исходной ширины к высоте (точное значение для расчета)
ratio = original_width / original_height

# Высота с округлением в меньшую сторону до целого (функция int() отсекает дробную часть)
target_height = int(target_width / ratio)

img_resized = cv2.resize(img_rgb, (target_width, target_height), interpolation=cv2.INTER_AREA)

# 3. Постройте гистограмму и найдите наибольшее значение плотности
# density=True возвращает оценку плотности вероятности (сумма значений * ширину бина = 1)
hist_r, _ = np.histogram(img_resized[:, :, 0].ravel(), bins=256, range=(0, 256), density=True)
hist_g, _ = np.histogram(img_resized[:, :, 1].ravel(), bins=256, range=(0, 256), density=True)
hist_b, _ = np.histogram(img_resized[:, :, 2].ravel(), bins=256, range=(0, 256), density=True)

max_density = max(np.max(hist_r), np.max(hist_g), np.max(hist_b))

# 4. Линейная нормировка и растяжение на диапазон 0-255 по каналам R, G, B
img_normalized = np.zeros_like(img_resized, dtype=np.float32)
for i in range(3):
    min_val = np.min(img_resized[:, :, i])
    max_val = np.max(img_resized[:, :, i])
    if max_val > min_val:
        img_normalized[:, :, i] = (img_resized[:, :, i] - min_val) / (max_val - min_val) * 255.0
    else:
        img_normalized[:, :, i] = 0.0
img_normalized = img_normalized.astype(np.uint8)

# Получение значения пикселя на пересечении 891 строки и 1146 столбца
# В Python индексация начинается с 0, поэтому 891 строка -> индекс 890, 1146 столбец -> индекс 1145
row_idx = 890
col_idx = 1145

if row_idx < img_normalized.shape[0] and col_idx < img_normalized.shape[1]:
    pixel_r = int(img_normalized[row_idx, col_idx, 0])
    pixel_g = int(img_normalized[row_idx, col_idx, 1])
    pixel_b = int(img_normalized[row_idx, col_idx, 2])
else:
    raise ValueError(f"Указанные координаты ({row_idx+1}, {col_idx+1}) выходят за границы изображения размером {img_normalized.shape[:2]}")

# 5. Вейвлет-преобразование Хаара для изображения (применяем поканально для R, G, B)
cA1_channels = []
cH1_channels = []
cV1_channels = []
cD1_channels = []

for i in range(3):
    cA, (cH, cV, cD) = pywt.dwt2(img_normalized[:, :, i].astype(float), 'haar')
    cA1_channels.append(cA)
    cH1_channels.append(cH)
    cV1_channels.append(cV)
    cD1_channels.append(cD)

# 6. Повторное вейвлет-преобразование Хаара для набора cA шага 5
cA2_channels = []
for i in range(3):
    cA2, _, _, _ = pywt.dwt2(cA1_channels[i], 'haar')
    cA2_channels.append(cA2)

# 7. Обнулите коэффициенты, меньшие по модулю значения threshold = 90, в наборе cA1
zero_count = 0
for i in range(3):
    mask = np.abs(cA1_channels[i]) < 90
    zero_count += np.sum(mask)
    cA1_channels[i][mask] = 0.0

# 8. Восстановление и отображение полученного изображения
img_rec_channels = []
for i in range(3):
    # Восстанавливаем из модифицированного cA1 и исходных cH1, cV1, cD1
    img_rec = pywt.idwt2((cA1_channels[i], (cH1_channels[i], cV1_channels[i], cD1_channels[i])), 'haar')
    img_rec_channels.append(img_rec)

img_final = np.stack(img_rec_channels, axis=-1)
img_final = np.clip(img_final, 0, 255).astype(np.uint8)

# Отображение результата
plt.figure(figsize=(10, 10))
plt.imshow(img_final)
plt.title("Восстановленное изображение после обнуления коэффициентов cA1")
plt.axis('off')
plt.show()

# Сохранение изображения (опционально, но рекомендуется по заданию)
cv2.imwrite('result_image.jpg', cv2.cvtColor(img_final, cv2.COLOR_RGB2BGR))

# ==============================================================================
# ВЫВОД ОТВЕТОВ ДЛЯ ЗАПОЛНЕНИЯ
# ==============================================================================
print("="*60)
print("ОТВЕТЫ ДЛЯ ЗАПОЛНЕНИЯ В ЗАДАНИИ:")
print(f"1. Коэффициент отношения ширины к высоте: {ratio:.3f}")
print(f"2. Высота полученного сжатого изображения: {target_height}")
print(f"3. Наибольшее значение плотности: {max_density:.4f}")
print(f"4. Интенсивность канала R: {pixel_r}")
print(f"5. Интенсивность канала G: {pixel_g}")
print(f"6. Интенсивность канала B: {pixel_b}")
print(f"7. Количество обнуленных коэффициентов в cA1: {zero_count}")
print("="*60)