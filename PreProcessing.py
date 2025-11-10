import os
import sys
import cv2
import numpy as np
from PIL import Image


def load_and_preprocess_image(input_path: str,
                              target_size: tuple = (256, 256),
                              clip_limit: float = 2.0,
                              tile_size: tuple = (8, 8)) -> np.ndarray:

    try:
        img_pil = Image.open(input_path)
        resized_img_pil = img_pil.resize(target_size, Image.Resampling.LANCZOS)

        img_np_rgb = np.array(resized_img_pil.convert('RGB'))
        img_bgr = cv2.cvtColor(img_np_rgb, cv2.COLOR_RGB2BGR)

        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        cl = clahe.apply(l)

        norm_channels = cv2.merge((cl, a, b))
        norm_img_bgr = cv2.cvtColor(norm_channels, cv2.COLOR_LAB2BGR)

        return norm_img_bgr

    except FileNotFoundError:
        print(f"ERROR: File {input_path} not found.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: An unexpected error occurred: {e}")
        sys.exit(1)
