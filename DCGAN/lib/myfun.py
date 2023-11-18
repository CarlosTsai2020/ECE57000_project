
from torchvision import transforms
import math
# write a function that edit a batch of image
# The code I wrote for stylegan is not reusable here. I am bad at coding QQ
FILL_VALUE = -0.95
def rotate(img, alpha, max_degree=45):
    return transforms.functional.rotate(img=img, angle=alpha * max_degree, fill = FILL_VALUE)

def shift(img, alpha, max_pixel=7):
    return transforms.functional.affine(img=img, translate=[alpha * max_pixel, 0], angle=0, scale=1, shear=0, fill = FILL_VALUE)

def zoom(img, alpha, max_scale=1.4):
    return transforms.functional.affine(img=img, translate=[0, 0], angle=0, scale=math.pow(max_scale, alpha), shear=0, fill = FILL_VALUE)


def get_edit_function(mode = 'r'):
    transformations = {'r': rotate, 's': shift, 'z': zoom}
    if mode not in transformations:
        raise ValueError("Invalid mode. Supported modes are 'r' (rotate), 's' (shift), and 'z' (zoom).")
    return transformations[mode]
