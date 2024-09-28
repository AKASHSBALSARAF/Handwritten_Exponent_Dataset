# Handwritten Exponent Dataset

This repository contains code for generating a dataset of handwritten digits with exponents, using the MNIST dataset as the base for digits and a custom dataset for exponents.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Directory Structure](#directory-structure)
- [Contributing](#contributing)
- [Research Paper](#research-paper)

## Introduction

This project generates a custom dataset of images combining MNIST digits with smaller exponent digits. The images can be used for training machine learning models to recognize mathematical expressions or similar tasks.

The dataset is generated by combining base digit images (from MNIST) with smaller exponent images, and the resulting images are saved along with a CSV file containing metadata.

## Dataset

The dataset consists of:

- **Base Images**: Digits from the MNIST dataset (0-9).
- **Exponent Images**: Custom exponent images stored in the `test` directory.
- **Combined Images**: Images where the base digits and exponent digits are combined, saved in the `OUTPUT` directory.

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/Handwritten-Exponent-Dataset.git
cd Handwritten-Exponent-Dataset
pip install -r requirements.txt

## Usage

1. Cropper Script:

The scripts/cropper.py script processes the MNIST dataset to create cropped images of the digits and save them in the data/EXPONENT directory.

python scripts/cropper.py

2. Combiner Script:

The scripts/combiner.py script combines the base digit images with exponent images and saves the resulting images in the data/OUTPUT directory.

python scripts/combiner.py

## Directory Structure

Handwritten_Exponent_Dataset/
│
├── data/
│   ├── EXPONENT/
│   │   ├── a1.png
│   │   ├── a2.png
│   │   ├── b1.png
│   │   ├── ...
│   │   └── image_data.csv
│   │
│   └── OUTPUT/
│       ├── img_1.png
│       ├── img_2.png
│       ├── ...
│       └── combined_image_data.csv
│
├── scripts/
│   ├── cropper.py
│   └── combiner.py
│
├── README.md
└── requirements.txt

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests.

## Research Paper

https://osf.io/preprints/osf/8jhtb

