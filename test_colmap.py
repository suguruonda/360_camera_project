import numpy as np
import pycolmap
from PIL import Image, ImageOps

image_path1 = "/multiview/datasets/360camera/test/monarch-001/images/monarch-001_id_001_130_000.JPG"
image_path2 = "/multiview/datasets/360camera/test/monarch-001/images/monarch-001_id_001_150_090.JPG"




# Input should be grayscale image with range [0, 1].
img = Image.open(image_path).convert('RGB')
img = ImageOps.grayscale(img)
img = np.array(img).astype(np.float) / 255.

# Optional parameters:
# - options: dict or pycolmap.SiftExtractionOptions
# - device: default pycolmap.Device.auto uses the GPU if available
sift = pycolmap.Sift()

# Parameters:
# - image: HxW float array
keypoints, scores, descriptors = sift.extract(img)
# Returns:
# - keypoints: Nx4 array; format: x (j), y (i), sigma, angle
# - scores: N array; DoG scores
# - descriptors: Nx128 array; L2-normalized descriptors