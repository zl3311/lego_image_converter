import os
import numpy as np
import requests
from PIL import Image
from basic_colormath import get_delta_e


def get_image_array(**kwargs):
    if 'url' in kwargs:
        assert 'filename' not in kwargs, 'Please only use either url or filename as input (not both).'
        response = requests.get(kwargs['url'], stream=True)
        img = Image.open(response.raw).convert('RGB')
        array = np.array(img)
    elif 'filename' in kwargs:
        assert os.path.exists('./upload/'+kwargs['filename']), 'File doesn\'t exist in ./upload/ folder! Please ' \
                                                               'double-check upload filename and location. '
        img = Image.open('./upload/'+kwargs['filename']).convert('RGB')
        array = np.array(img)
    else:
        raise Exception(
            'Please use either url to fetch an image from Internet or upload a local image to /upload folder and use '
            'it as filename.')
    return array


def match_color(rgb, block_colors):
    return sorted(
        [(get_delta_e(rgb, k), k) for k in block_colors]
        , key=lambda x: x[0]
    )
