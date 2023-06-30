# Junction Density Analysis

🙌 Junction Density Analysis using [OpenCV](https://opencv.org) and [YOLOv5](https://github.com/ultralytics/yolov5)!

## Overview

💡 Vehicle and human detection were made through bird's-eye view images recorded with an unmanned aerial vehicle camera. The object counter is integrated into the YOLO algorithm, which is trained with a special data set. Density parameters are returned over the predetermined number of objects according to the intersection size.

<p align="center">
  <img src="https://i.hizliresim.com/kwsggha.gif" />
  <br>Junction Density Analysis Using OpenCV and YOLOv5
</p>

## Installations ⬇️

✔️ A virtual environment is created for the system. (Assuming you have [Anaconda](https://www.anaconda.com/) installed.)

```bash
conda create -n yolov5_junction python -y
conda activate yolov5_junction
```

✔️ Clone repo and install [requirements.txt](https://github.com/zahidesatmutlu/yolov5-sahi/blob/master/requirements.txt) in a [Python>=3.7.0](https://www.python.org/downloads/) (3.9 recommended) environment, including [PyTorch>=1.7](https://pytorch.org/get-started/locally/) (1.9.0 recommended).

```bash
git clone https://github.com/zahidesatmutlu/Junction-Density-Analysis  # clone
cd Junction-Density-Analysis
pip install -r requirements.txt  # install
```

✔️ Install [CUDA Toolkit](https://developer.nvidia.com/cuda-11-6-0-download-archive) version 11.6 and install [PyTorch](https://pytorch.org/get-started/previous-versions/) version 1.9.0.

```bash
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio===0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```


✔️ The file structure should be like this:

```bash
Junction-Density-Analysis/
    .gitattributes
    best.pt
    detect.py
    project.mp4
    requirements.txt
```

## Usage 🔷

```python
.
.
.
x = 0
# Print results
for c in det[:, -1].unique():
    n = (det[:, -1] == c).sum()  # detections per class
    x += n
    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
    xs = f"{x} cars detected"
    ys = f"{x} cars detected"
    if x <= 7:
        ys = f"Junction density: Low"
    elif x > 7 and x < 11:
        ys = f"Junction density: Medium"
    elif x >= 11:
        ys = f"Junction density: High"
.
.
.
.

# Stream results
im0 = annotator.result()
if view_img:
    if platform.system() == 'Linux' and p not in windows:
        windows.append(p)
        cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
        cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
    cv2.imshow(str(p), im0)
    cv2.waitKey(1)  # 1 millisecond
cv2.putText(im0, xs, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
cv2.putText(im0, ys, (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
.
.
.
```

## Resources 🤝

🔸 [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)

🔸 [https://opencv.org](https://opencv.org)
