# Deepfake Detection using Explainable AI (XAI)

## Overview

This project implements a **deepfake detection system** using deep learning models and integrates **Explainable AI (XAI)** techniques to provide transparency in predictions. The system analyzes images/videos to classify them as *real* or *fake* and highlights manipulated regions.

---

## Features

* Deepfake detection using CNN-based models
* Frame extraction from videos
* Performance evaluation using standard metrics
* Visual explanations using:

  * Grad-CAM
  * SHAP
* Streamlit-based web interface for user interaction

---

## Project Structure

```
├── extract_frames.py        # Extract frames from video
├── train_on_frames.py       # Model training script
├── performance_metrics.py   # Evaluation metrics
├── video_app.py             # Streamlit app for detection
├── README.md                # Project documentation
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/thanusreesomar/Deepfake-detection-using-xai.git
cd Deepfake-detection-using-xai
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Requirements

Create a `requirements.txt` file with:

```
torch
torchvision
streamlit
opencv-python
matplotlib
numpy
pillow
shap
pytorch-grad-cam
```

---

## Usage

### Run the Streamlit App

```bash
streamlit run video_app.py
```

### Steps

1. Upload image/video
2. System processes input
3. Displays prediction (Real/Fake)
4. Shows explanation heatmap (XAI)

---

## Model Details

* Architecture: Convolutional Neural Network (CNN)
* Input: Extracted video frames / images
* Output: Binary classification (Real vs Fake)

---

## Explainable AI Techniques

* **Grad-CAM**: Highlights important regions influencing prediction
* **SHAP**: Explains feature contributions

---

## Performance Metrics

* Accuracy
* Precision
* Recall
* F1-score

---

## Applications

* Fake news detection
* Social media content verification
* Digital forensics
* Identity verification systems

---

## Future Work

* Improve model generalization
* Add real-time detection
* Support multimodal inputs (audio + video)
* Deploy as web/cloud service

---

## Author

**Thanu Sree Somar**

---

## License

This project is for academic and research purposes.
