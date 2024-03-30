import io
import time

import cv2 as cv
import numpy as np

from matplotlib import pyplot as plt
from skimage import transform
from ultralytics import YOLO

from display import PlotDisplay


def figure_to_array(fig):
    fig.canvas.draw()
    return np.array(fig.canvas.renderer._renderer)


vid = cv.VideoCapture("footage_full.mp4")
if not vid.isOpened():
    print("Error opening video stream or file")
    exit()

ret, frame = vid.read()
frame_height, frame_width, _ = frame.shape

img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
background = cv.imread("footage_test_bg.png")

img_plot = PlotDisplay(img)
plt.show(block=True)
background_plot = PlotDisplay(background)
plt.show(block=True)

src = np.array(img_plot.line_builder.get_points()).reshape(4, 2)
dst = np.array(background_plot.line_builder.get_points()).reshape(4, 2)

model = YOLO('yolov8n.pt')

fourcc = cv.VideoWriter_fourcc(*'mp4v')
output = cv.VideoWriter('video.avi', fourcc, 30, (500, 500))

while True:
    ret, frame = vid.read()
    if not ret:
        break

    img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = model(img, save=False, classes=0, project=r"D:\resources\cs\sem4\dip\3D_Tracking")

    tform = transform.estimate_transform('projective', src, dst)
    tf_img = transform.warp(img, tform.inverse)

    people = []
    boxes = results[0].boxes.xyxy.tolist()

    for box in boxes:
        midpoint = ((box[0] + box[2]) / 2, box[3])
        homogenous_coords = np.dot(tform, np.array([*midpoint, 1]).reshape(3, 1)).flatten()
        x_prime, y_prime, w_prime = homogenous_coords.tolist()

        # convert back to cartesian
        people.append((x_prime / w_prime, y_prime / w_prime))

    plt.figure(figsize=(5, 5))
    for point in people:
        plt.scatter(*point)

    plt.imshow(background)
    plt.grid(False)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    output.write(cv.imdecode(np.frombuffer(buf.getvalue(), dtype=np.uint8), 1))
    plt.close()

cv.destroyAllWindows()
output.release()
