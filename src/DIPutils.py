# imporing necessary libraries
import cv2
import numpy as np

# function to compress an image
def compress_image(image, target_size=500):
    h, w = image.shape[:2]
    if h > w:
        new_h, new_w = target_size, int((w / h) * target_size)
    else:
        new_w, new_h = target_size, int((h / w) * target_size)
        
    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image

# function to remove noise from an image
def image_denoising(image, denoising = 3, color_denoising = 3):
    image = cv2.fastNlMeansDenoisingColored(image, None, denoising, color_denoising, 7, 21)
    return image

# function to sharpen an image
def image_sharpening(image , kernel = np.array([[0, -1, 0],[-2, 8, -2],[0, -2, 0]])):
    image = cv2.filter2D(image, -1, kernel)
    return image

# function for radiometric calibration of an image
def radiometric_calibration(image, gain=1.3):
    image_corrected = image / gain
    return np.clip(image_corrected, 0, 255).astype(np.uint8)

# function for image segmentation
def image_segmentation(img):
    ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    markers = cv2.connectedComponents(sure_fg)[1]
    markers += 1
    markers[unknown == 255] = 0
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.watershed(img_color, markers)
    img_color[markers == -1] = [0, 255, 0]
    return img_color

# function to detect edges in an image
def image_edges(image, type):
    # best == canny
    if(type == 'canny'):
        image = cv2.Canny(image, 200, 400)
    elif(type == 'sobel'):
        image = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    elif(type == 'laplacian'):
        image = cv2.Laplacian(image, cv2.CV_64F)
    return image