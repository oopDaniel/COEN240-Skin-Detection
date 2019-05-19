import sys
import os
from math import exp, sqrt
from PIL import Image
import numpy as np

DETECTED_FILE_NAME = "./detected.jpg"
DETECTED_FILE_NAME_WITHOUT_EXT = "./detected"

def read_image_file(file_path, png=False):
    """
    Read and parse the file into image. `file_path` should be something like
    'data/image-name' without extension, and the extension can be assigned as '.jpg'
    or '.png'. Returns the image.
    """
    ext = '.png' if png else '.jpg'
    im = Image.open(file_path + ext, 'r')
    return im.convert('RGB')

def img_to_pixel(img):
    """
    Convert image into a list of (r, g, b) tuple pixels
    """
    return list(img.getdata())

def is_skin_by_mask(mask_pix):
    """
    Determine the while pixel as skin using mask image
    """
    r, g, b = mask_pix
    return r > 250 and g > 250 and b > 250

def train_image(img_pix, mask_img):
    """
    Calculate the probability for pixels based on image and mask
    """
    img_pix, mask_pix = img_to_pixel(img), img_to_pixel(mask_img)

    N = len(img_pix)
    r_ratio_skin, r_ratio_non_skin = [], []
    g_ratio_skin, g_ratio_non_skin = [], []
    skin_count = 0

    for i in range(N):
        r, g, b = img_pix[i]
        rgb = r + g + b
        r_ratio = r / rgb if rgb > 0 else 1 / 3
        g_ratio = g / rgb if rgb > 0 else 1 / 3

        if (is_skin_by_mask(mask_pix[i])):
            skin_count += 1
            r_ratio_skin.append(r_ratio)
            g_ratio_skin.append(g_ratio)
        else:
            r_ratio_non_skin.append(r_ratio)
            g_ratio_non_skin.append(g_ratio)

    # Calculate H0 / H1
    skin_probability = skin_count / len(img_pix)
    non_skin_probability = 1 - skin_probability

    # Convert list into np list
    r_ratio_skin, r_ratio_non_skin = np.array(r_ratio_skin), np.array(r_ratio_non_skin)
    g_ratio_skin, g_ratio_non_skin = np.array(g_ratio_skin), np.array(g_ratio_non_skin)

    # Calculate mean and variance
    r_mean_skin, g_mean_skin = np.mean(r_ratio_skin), np.mean(g_ratio_skin)
    r_mean_non_skin, g_mean_non_skin = np.mean(r_ratio_non_skin), np.mean(g_ratio_non_skin)
    r_var_skin, g_var_skin = np.var(r_ratio_skin), np.var(g_ratio_skin)
    r_var_non_skin, g_var_non_skin = np.var(r_ratio_non_skin), np.var(g_ratio_non_skin)

    return {
        "r_mean_skin": r_mean_skin,
        "r_mean_non_skin": r_mean_non_skin,
        "g_mean_skin": g_mean_skin,
        "g_mean_non_skin": g_mean_non_skin,
        "r_var_skin": r_var_skin,
        "r_var_non_skin": r_var_non_skin,
        "g_var_skin": g_var_skin,
        "g_var_non_skin": g_var_non_skin,
        "skin_probability": skin_probability,
        "non_skin_probability": non_skin_probability,
        "threshold": non_skin_probability / skin_probability
    }

def detect_skin(img, params):
    """
    Convert image into a list of (r, g, b) tuple pixels
    """
    skin_img = Image.new(img.mode, img.size)
    width, height = img.size
    threshold = params["threshold"]
    r_mean_skin, r_mean_non_skin = params["r_mean_skin"], params["r_mean_non_skin"]
    g_mean_skin, g_mean_non_skin = params["g_mean_skin"], params["g_mean_non_skin"]
    r_var_skin, r_var_non_skin = params["r_var_skin"], params["r_var_non_skin"]
    g_var_skin, g_var_non_skin = params["g_var_skin"], params["g_var_non_skin"]

    def is_skin_by_probability(pixel):
        """
        Determine if it's skin using the parameters of Bayesian classifier model
        from previous training image.
        """
        r, g, b = pixel
        rgb = r + g + b
        r_ratio = r / rgb if rgb > 0 else 1 / 3
        g_ratio = g / rgb if rgb > 0 else 1 / 3
        probability_kth_given_skin = exp(-(r_ratio - r_mean_skin) ** 2 / (2 * r_var_skin) - (g_ratio - g_mean_skin) ** 2 / (2 * g_var_skin)) / (2 * np.pi * sqrt(r_var_skin) * sqrt(g_var_skin))
        probability_kth_given_non_skin = exp(-(r_ratio - r_mean_non_skin) ** 2 / (2 * r_var_non_skin) - (g_ratio - g_mean_non_skin) ** 2 / (2 * g_var_non_skin)) / (2 * np.pi * sqrt(r_var_non_skin) * sqrt(g_var_non_skin))
        prediction = probability_kth_given_skin / probability_kth_given_non_skin
        return prediction >= threshold

    for y in range(height):
        for x in range(width):
            if (is_skin_by_probability(img.getpixel((x, y)))):
                skin_img.putpixel((x, y), (255, 255, 255))
            else:
                skin_img.putpixel((x, y), (0, 0, 0))

    skin_img.save(DETECTED_FILE_NAME)

def calc_accuracy(img, mask_img):
    """
    Calculate the accuracy (true positive, true negative, false positive, etc.)
    """
    img_pix, mask_pix = img_to_pixel(img), img_to_pixel(mask_img)

    N = len(img_pix)
    skin_count = 0
    true_positive_count, true_negative_count = 0, 0
    for i in range(N):
        if (is_skin_by_mask(mask_pix[i])):
            skin_count += 1
            if (is_skin_by_mask(img_pix[i])):
                true_positive_count += 1
        else:
            if (not is_skin_by_mask(img_pix[i])):
                true_negative_count += 1

    true_positive_rate = true_positive_count / skin_count
    true_negative_rate = true_negative_count / (N - skin_count)
    false_positive_rate = 1 - true_negative_rate
    false_negative_rate = 1 - true_positive_rate

    return {
        "true_positive_rate": true_positive_rate,
        "true_negative_rate": true_negative_rate,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
    }

def print_accuracy(accuracy):
    print("True Positive Rate:", str(accuracy["true_positive_rate"] * 100) + "%")
    print("True Negative Rate:", str(accuracy["true_negative_rate"] * 100) + "%")
    print("False Positive Rate:", str(accuracy["false_positive_rate"] * 100) + "%")
    print("False Negative Rate:", str(accuracy["false_negative_rate"] * 100) + "%")

if __name__ == '__main__':
    training_path = sys.argv[1]
    test_path = sys.argv[2]

    # Parse raw data
    img, mask_img = read_image_file(training_path), read_image_file(training_path, True)

    # Calculate required params with data
    params = train_image(img, mask_img)

    # Parse test img
    test_img, test_mask_img = read_image_file(test_path), read_image_file(test_path, True)

    # Perform skin detection
    detect_skin(test_img, params)

    # Calculate accuracy
    res_img = read_image_file(DETECTED_FILE_NAME_WITHOUT_EXT)
    accuracy = calc_accuracy(res_img, test_mask_img)
    print_accuracy(accuracy)

    # Cleanup
    img.close()
    mask_img.close()
    test_img.close()
    test_mask_img.close()
    res_img.close()