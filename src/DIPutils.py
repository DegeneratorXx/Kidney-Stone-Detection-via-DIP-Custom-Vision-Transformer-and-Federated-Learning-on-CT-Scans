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
def image_denoising(image, denoising = 5, color_denoising = 5):
    denoised = cv2.fastNlMeansDenoising(image, None, denoising)
    return denoised

# function to sharpen an image
def image_sharpening(image , kernel = np.array([[0, -1, 0],[-2, 8, -2],[0, -2, 0]])):
    image = cv2.filter2D(image, -1, kernel)
    return image

# function for radiometric calibration of an image
def radiometric_calibration(image, gain=1.3):
    image_corrected = image / gain
    return np.clip(image_corrected, 0, 255).astype(np.uint8)

class DIPTransform:
    def __init__(self, target_size=500):
        self.target_size = target_size

    def __call__(self, img):
        # Convert PIL image to NumPy array (OpenCV works with NumPy arrays)
        img = np.array(img)

        # Apply DIP functions
        img = compress_image(img, target_size=self.target_size)
        img = image_denoising(img)
        img = image_sharpening(img)
        img = radiometric_calibration(img)

        # Convert back to PIL image (to ensure compatibility with PyTorch transforms)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV uses BGR, convert to RGB
        return img