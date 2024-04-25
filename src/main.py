import io

import cv2 as cv
import numpy as np

from deep_sort_realtime.deepsort_tracker import DeepSort
from matplotlib import pyplot as plt
from skimage import transform
from ultralytics import YOLO

from display import PlotDisplay


vid = cv.VideoCapture("footage.mp4")
if not vid.isOpened():
    print("Error opening video stream or file")
    exit()

ret, frame = vid.read()
frame_height, frame_width, _ = frame.shape

img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
background = cv.imread("bg.png")

img_plot = PlotDisplay(img)
plt.show(block=True)
background_plot = PlotDisplay(background)
plt.show(block=True)

src = np.array(img_plot.line_builder.get_points()).reshape(4, 2)
dst = np.array(background_plot.line_builder.get_points()).reshape(4, 2)
tform = transform.estimate_transform('projective', src, dst)

model = YOLO('yolov8n.pt')
tracker = DeepSort(max_age=7, n_init=2, max_cosine_distance=0.3)

fourcc = cv.VideoWriter_fourcc(*'mp4v')
output = cv.VideoWriter('video.avi', fourcc, vid.get(cv.CAP_PROP_FPS), (500, 500))

plt.figure(figsize=(5, 5))
plt.tight_layout()

i = 0
while vid.isOpened():
    ret, frame = vid.read()
    if not ret:
        break

    img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = model(img, save=False, classes=0, conf=0.4)

    people = []
    boxes = results[0].boxes.xyxy.tolist()
    boxes_ltwh = []

    for box in results[0].boxes:
        xyxy = box.xyxy.tolist()[0]
        conf = box.conf
        x_min, y_min, x_max, y_max = xyxy[0], xyxy[1], xyxy[2], xyxy[3]

        left = x_min
        top = y_min
        width = x_max - x_min
        height = y_max - y_min

        boxes_ltwh.append(([left, top, width, height], conf, 'person'))

    tracks = tracker.update_tracks(boxes_ltwh, frame=img)

    for track in tracks:
        track_id = track.track_id
        box = track.to_ltrb()
        
        midpoint = ((box[0] + box[2]) / 2, box[3])
        homogenous_coords = np.dot(tform, np.array([*midpoint, 1]).reshape(3, 1)).flatten()
        x_prime, y_prime, w_prime = homogenous_coords.tolist()

        # convert back to cartesian
        people.append(((x_prime / w_prime, y_prime / w_prime), track_id))

    plt.imshow(background)
    plt.axis('off')
    plt.grid(False)

    for point, track_id in people:
        plt.scatter(*point, color='red')
        plt.text(*point, track_id)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.clf()
    buf.seek(0)

    mapped_2d = cv.imdecode(np.frombuffer(buf.getvalue(), dtype=np.uint8), 1)
    mapped_2d = cv.resize(mapped_2d, (mapped_2d.shape[1], int(mapped_2d.shape[0] * (max(mapped_2d.shape[0], img.shape[0]) / mapped_2d.shape[0]))), interpolation=cv.INTER_CUBIC)
    output_frame = cv.hconcat([mapped_2d, img[:, :, ::-1]])

    cv.imshow('Skynet', output_frame)
    i += 1

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()
output.release()
