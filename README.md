
---

# Violence Detection

##  Overview

**Violence Detection** is a real-time object detection system designed to identify violent elements in images and videos. Using **Ultralytics YOLOv11** and a custom dataset managed via **Roboflow**, the model can accurately detect instances of:

*    Blood
*    Weapon
*    General Violence

---

##  Features

* **Multi-Class Detection**: Detects `blood`, `weapon`, and `violence` in real-time.
* **Fast & Accurate**: Powered by YOLOv11 for high-performance object detection.
* **Custom Dataset**: Curated and preprocessed with Roboflow for optimal model training.
* **Transfer Learning**: Fine-tuned YOLO model trained on domain-specific images.

---

## Tools 
* **Ultralytics YOLOv8** (object detection)
* **Roboflow** (dataset annotation & preprocessing)


---
## Tech Stack
* **Python**
* **PyTorch**
* **OpenCV**
* **Streamlit**
---

## Dataset
* Annotated using **Roboflow**
* Classes: `Blood`, `Weapon`, `Violence`,`NonViolence`
* Includes image augmentation and preprocessing steps for better generalization
---

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/violence-detection.git
cd violence-detection

# Install dependencies
pip install -r requirements.txt
```




