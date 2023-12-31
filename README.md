
# Tumor Detection on Brain MRI using Detectron2

This project demonstrates the application of computer vision in healthcare by detecting tumors 
in brain MRI images. The project utilizes the Detectron2 framework, focusing on the 
COCO-Detection/retinanet baseline model, to achieve accurate tumor detection. Additionally, the 
project is presented as a web application using Streamlit for user-friendly interaction.
![](https://github.com/sevdaimany/Tumor-Detection-Brain-MRI/blob/master/screenshot.png)


## Table of Contents
- [Detectron2 Overview](#detectron2-overview)
- [COCO-Detection/retinanet Baseline](#retinanet-baseline)
- [Web Application](#web-application)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Web Application](#web-application)

## Detectron2 Overview

Detectron2 is a cutting-edge object detection framework developed by Facebook AI Research 
(FAIR). It simplifies the process of building, training, and deploying object detection models. 
By providing modular components and pre-implemented state-of-the-art algorithms, Detectron2 
enables efficient experimentation and development in the field of computer vision.
## Retinanet Baseline

The COCO-Detection/retinanet baseline is a specific object detection model architecture 
available within Detectron2. RetinaNet is a one-stage object detection model known for its 
efficient speed and competitive accuracy. It's especially suited for detecting objects of 
varying scales and sizes, making it a suitable choice for medical image analysis like tumor 
detection in brain MRI images. The COCO-Detection baseline further fine-tunes the RetinaNet 
architecture on the COCO dataset, which is a widely-used benchmark for object detection tasks.
## Web Application

The user-friendly web application powered by Streamlit simplifies the utilization of the tumor 
detection model. Users can effortlessly upload their brain MRI images and instantly visualize 
the detected tumor regions. 


## Dataset
The dataset used in this project is sourced from [Roboflow 
Universe](https://universe.roboflow.com/tfg-2nmge/axial-dataset). This dataset consists of 
axial brain MRI images annotated with tumor regions.

## Project Structure

* **train**: This directory contains files for training the model. The trained model weights 
are saved in the `/train/output/`directory, eliminating the need for retraining. The project's 
training progress, such as loss and validation plots, can be found here as well. The selected 
model weights for deployment were taken from epoch number 3000 based on the provided training 
and validation plots.
![](https://github.com/sevdaimany/Tumor-Detection-Brain-MRI/blob/master/train/train_val_loss.png)
## Installation
To set up and run the Tumor Detection project locally, follow these steps:

 Clone the repository:

```bash
git clone https://github.com/sevdaimany/Tumor-Detection-Brain-MRI
cd Tumor-Detection-Brain-MRI
```
Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
Install the required dependencies:

```bash
pip install -r requirements.txt
```

Download the dataset from the provided link and place the images in the appropriate 
data/train/imgs, data/train/anns, data/val/imgs and data/val/anns folders.

Download the trained model dataset from [here](https://drive.google.com/file/d/1-BfYQ5X7UdGQXgy1smfpI1oylWuxeuVW/view?usp=sharing) and put it in train/output folder.
## Web Application

The web application provides a user-friendly interface for tumor detection:

Launch the Streamlit app:

```bash
streamlit run main.py
```
Open your web browser and navigate to the provided URL.

Upload an MRI image using the app's interface.

View the uploaded image with highlighted tumor regions.
