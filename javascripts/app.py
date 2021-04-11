import tkinter as tk
import tkinter as ttk
from tkinter import *
from tkinter import filedialog, Text
import os
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.image as mpimg
import cv2
import scipy
from skimage.filters import threshold_otsu
from skimage import color
from skimage import io
import numpy as np
from tkinter.font import Font
import csv
from skimage.filters import threshold_local
from skimage.filters import try_all_threshold
from skimage.filters import sobel
from skimage.filters import gaussian
from skimage import exposure
from skimage import morphology
import os
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from skimage.restoration import inpaint
from skimage.feature import Cascade
from skimage import data
from skimage.segmentation import slic
from  skimage.color import label2rgb
from skimage.filters import gaussian
from skimage import io
from skimage.feature import Cascade
from skimage import data
from flask import Flask
from flask import request
from flask import render_template
from flask_cors import CORS, cross_origin

app = Flask(__name__)
@app.route("/re" , methods=['GET'])

def re():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    cwd = os.getcwd()
    print(cwd)
    os.chdir(dir_path)
    image1=""
    image = io.imread("somefilename.jpg")
    # Load the trained file from data
    trained_file = data.lbp_frontal_face_cascade_filename()

    # Initialize the detector cascade
    detector = Cascade(trained_file)
    detected = detector.detect_multi_scale(
        img=image,
        scale_factor=1.2,
        step_ratio=1,
        min_size=(50, 50),
        max_size=(100, 100)
    )
    resulting_image=image
    for detected_face in detected:
        face = get_face(image, detected_face)
        blurred_face = gaussian(face, multichannel=True, sigma=10)
        #show_image(face)
        resulting_image = merge_blurry_face(image, detected_face, blurred_face)
    show_image(resulting_image, "Blurred faces")
    os.remove("somefilename.jpg");
    return 'Its Done!'

def show_image(image, title='Image', cmap_type='gray'):
    dpi = 80
    im_data = plt.imread('somefilename.jpg')
    height, width, nbands = im_data.shape
    figsize = width / float(dpi), height / float(dpi)
    fig = plt.figure(figsize=figsize)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(image, interpolation='nearest')
    ax.set(xlim=[-0.5, width - 0.5], ylim=[height - 0.5, -0.5], aspect=1)
    fig.savefig("foo.jpg", dpi=dpi,transparent=True)

def get_face(image, detected_face):
    x_from, y_from = detected_face['r'], detected_face['c']
    x_to = x_from + detected_face['width']
    y_to = y_from + detected_face['height']
    face = image[x_from:x_to, y_from:y_to]
    return face

def merge_blurry_face(original, detected_face, gaussian_face):
    x_from, y_from = detected_face['r'], detected_face['c']
    x_to = x_from + detected_face['width']
    y_to = y_from + detected_face['height']
    original[x_from:x_to, y_from:y_to] = 255 * gaussian_face
    return original

# Import the corner detector related functions and modul

if __name__ == '__main__':
    app.run(port=5000, debug=True)
