# Fashion Recommendation System

## Overview

This fashion recommendation system provides personalized fashion item recommendations. Users can search for products by text or upload an image of a fashion item to find similar products. Additionally, the system features a virtual try-on option where users can upload their image to see how different clothes would look on them.

## Features

- **Text Search**: Enter descriptive text to find fashion items that match the description.
- **Image Upload**: Upload an image of a fashion item to find similar products.
- **Virtual Try-On**: Upload a personal image to virtually try on recommended fashion items.

## How It Works

The recommendation engine uses a pre-trained EfficientNet B7 model to extract features from fashion item images. It then compares these features using a K-Nearest Neighbors algorithm to find the most similar items from our dataset. For text queries, it uses the CLIP model to encode the text and matches it with image features.

## Virtual Try-On

Utilizing RapidAPI's Virtual Try-On API, the system overlays recommended clothing items onto the user's uploaded image, allowing for a virtual fitting room experience.

## Prerequisites

- Git
- Python 3.8 or higher
- pip (Python package installer)
- virtualenv (optional but recommended)

## Setup Guide

### Step 1: Clone the Repository

Clone the project repository to your local machine by running the following command:

```bash
git clone https://github.com/priyamthakkar2001/Fashion_Recommendation_and_VirtualTryon.git
cd "Fashion_Recommendation_and_VirtualTryon/Fashion Project Flask"
```

### Step 2: Create and Activate a Virtual Environment

Create a virtual environment to manage the project's dependencies separately from your global Python setup:

```bash
python -m venv venv
source venv/bin/activate  # Use `venv\Scripts\activate` on Windows
```

### Step 3: Install Required Packages

Install all the necessary Python packages specified in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Step 4: Download the Dataset

Use `wget` to download the dataset required for the application. If `wget` is not installed on your system, follow the installation instructions based on your operating system.

Install wget
- For Ubuntu/Debian:

```bash
sudo apt install wget
```

- For Fedora::

```bash
sudo dnf install wget
```

- For macOS:

```bash
brew install wget
```

Download and Extract the Dataset

```bash
wget -O images.zip "https://drive.google.com/uc?export=download&id=1gvfLUIq-xbMadJ3X1TEUqvYVQuTry5Dx"
unzip images.zip
```

Ensure you are in the "Fashion Project Flask" directory when you unzip to keep all files organized correctly.

### Step 5: Setup Virtual Try-On Feature

To use the Virtual Try-On feature, follow these steps:

1. Subscribe to the API:

   - Visit Virtual Try-On API on RapidAPI
   - Subscribe to the service (free for 3 Virtual Try-Ons, with each Try-On allowing up to 5 recommended images).
   - Copy your API key from the API dashboard.

2. Configure API Key:

   - Open app.py in your text editor.
   - Navigate to line 116 and replace "your_rapidapi_key" with the API key you copied:

```bash
'X-RapidAPI-Key': "paste_your_api_key_here"
```

### Step 6: Run the Application

With the dataset in place and the environment set up, start the Flask application:

```bash
python app.py
```

The application will run on http://127.0.0.1:5000/ by default. You can access it using any web browser.


## Usage

Navigate to `http://127.0.0.1:5000/` in your web browser to start exploring the Fashion Recommendation System. Use the features provided to view different fashion styles and try them on virtually.

## Conclusion

By following these detailed instructions, you should be able to set up and start using the Fashion Recommendation System without any problems. For additional support or to report issues, use the project's GitHub issues page.

## Team

Priyam Thakkar(pt50)

Anupreet Sihra(as368)

Joseph Mohanty(jm215)
