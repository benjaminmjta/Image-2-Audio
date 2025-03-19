import numpy as np
from PIL import Image
import sys

def compare_images(image_path1, image_path2):
    """
    compares 2 images pixel by pixel.
    :param image_path1: path of the first image
    :param image_path2: path of the second image
    """
    img1 = Image.open(image_path1).convert("RGB")
    img2 = Image.open(image_path2).convert("RGB")

    if img1.size != img2.size:
        print("error - the images are not the same size")
        return

    img1_array = np.array(img1)
    img2_array = np.array(img2)

    total_pixels = img1_array.size // 3
    identical_pixels = np.sum(np.all(img1_array == img2_array, axis=-1))

    similarity = (identical_pixels / total_pixels) * 100

    print(f"result: {similarity:.2f}% ({identical_pixels} of {total_pixels} pixel identical)")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("please pass 2 image paths.")
        print("example: python compare.py image1.png image2.png")
    else:
        compare_images(sys.argv[1], sys.argv[2])