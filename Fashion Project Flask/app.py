from flask import Flask, request, render_template, send_from_directory, url_for
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import EfficientNetB7
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
import os
import pandas as pd
import torch
import clip
from PIL import Image
import pickle
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors
import requests
from flask import send_file
import os
import requests
from flask import Flask, request, render_template, send_from_directory, url_for
from werkzeug.utils import secure_filename
import os



app = Flask(__name__)

# app.config['IMAGES_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images')




# Configure paths for the data and uploads
UPLOAD_FOLDER = 'uploads'
DATA_FOLDER = 'data'
IMAGES_FOLDER = 'images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['IMAGES_FOLDER'] = IMAGES_FOLDER


# Initialize the EfficientNet B7 model for image processing
base_model = EfficientNetB7(weights='imagenet', include_top=False, input_shape=(600, 600, 3))
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
image_model = Model(inputs=base_model.input, outputs=x)

# Load precomputed data for image recommendations
feature_list = np.array(pickle.load(open(os.path.join(DATA_FOLDER, 'embeddings_EfficientB7.pkl'), 'rb')))
filenames = pickle.load(open(os.path.join(DATA_FOLDER, 'filenames.pkl'), 'rb'))
neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
neighbors.fit(feature_list)

# Initialize the CLIP model for text-to-image search
device = "mps" if torch.backends.mps.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)
styles_df = pd.read_csv(os.path.join(DATA_FOLDER, 'styles.csv'), on_bad_lines='skip')
# styles_df['image_path'] = styles_df['id'].apply(lambda x: os.path.join(IMAGES_FOLDER, f'{x}.jpg'))
styles_df['image_path'] = styles_df['id'].apply(lambda x: f'{x}.jpg')

image_features = pickle.load(open(os.path.join(DATA_FOLDER, 'image_features.pkl'), 'rb'))

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        search_text = request.form.get('search_text')
        product_image = request.files.get('file')
        person_image = request.files.get('tryon_image')

        tryon_images = []
        product_info = []
        recommendations = []

        if person_image:
            person_image_path = save_image(person_image)  # Save and get path of the person's image

        if search_text or product_image:
            recommendations = recommend(search_text) if search_text else process_image_file(product_image)
            for cloth_filename in recommendations:
                cloth_id = cloth_filename.split('/')[-1].split('.')[0]
                product_dict = styles_df.loc[styles_df['id'] == int(cloth_id)].to_dict('records')[0]
                product_info.append(product_dict)
                if person_image:
                    tryon_result = virtual_tryon(person_image_path, cloth_filename)
                    tryon_images.append(tryon_result if tryon_result else {})

            if not tryon_images:  # Ensures there's a placeholder if no tryon images are processed
                tryon_images = [{}] * len(recommendations)

            return render_template('results.html', recommendations=zip(recommendations, tryon_images, product_info), tryon=bool(person_image))

    return render_template('index.html')



def save_image(image):
    filename = secure_filename(image.filename)
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image.save(image_path)
    return image_path

def virtual_tryon(person_image_path, cloth_image_filename):
    # cloth_image_path = os.path.join(app.config['IMAGES_FOLDER'], cloth_image_filename)
    cloth_image_path = cloth_image_filename[1:]
     # Ensure the files exist before attempting to open them
    if not os.path.exists(cloth_image_path) or not os.path.exists(person_image_path):
        print(f"File not found: {cloth_image_path} or {person_image_path}")
        return None

    with open(person_image_path, 'rb') as person_file, open(cloth_image_path, 'rb') as cloth_file:
        files = {
            'personImage': ('person.jpg', person_file),
            'clothImage': ('cloth.jpg', cloth_file)
        }
        headers = {
            'X-RapidAPI-Key': "your_rapidapi_key",
            'X-RapidAPI-Host': "virtual-try-on2.p.rapidapi.com"
        }
        
        response = requests.post(
            'https://virtual-try-on2.p.rapidapi.com/clothes-virtual-tryon',
            headers=headers,
            files=files
        )

        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                # Corrected the key based on the actual API response
                return data['response']['ouput_path_img']
            else:
                print("API did not return a success response.")
        else:
            print(f"API request failed with status code: {response.status_code}")

        # Print the full response if there's an error
        print(f"API response: {response.text}")
        return None

def process_image_file(uploaded_file):
    if uploaded_file and uploaded_file.filename != '':
        filename = secure_filename(uploaded_file.filename)
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        uploaded_file.save(img_path)
        result = process_single_image(img_path)
        
        distances, indices = neighbors.kneighbors([result])
        recommendations = []
        for index in indices[0][1:]:  # Skip the first index (it's the query image itself)
            try:
                image_id = filenames[index].split('/')[-1].split('.')[0]
                image_path = styles_df.loc[styles_df['id'] == int(image_id), 'image_path'].values[0]
                recommendations.append(url_for('uploaded_file', filename=image_path))
            except IndexError as e:
                print(f"Index error: {e}, index was {index}")
            except Exception as e:
                print(f"An error occurred: {e}")

        # Clean up the uploaded image if needed
        os.remove(img_path)

        return recommendations

    return []



def process_single_image(img_path):
    img = image.load_img(img_path, target_size=(600, 600))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocess_img = preprocess_input(expanded_img_array)
    features = image_model.predict(preprocess_img).flatten()
    normalized_features = features / norm(features)
    return normalized_features

def recommend(text_query):
    text_features = encode_text(text_query).cpu().numpy().flatten()
    similarities = []
    for product_id, image_feature in image_features.items():
        similarity = 1 - distance.cosine(text_features, image_feature)
        similarities.append((product_id, similarity))
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_indices = [url_for('uploaded_file', filename=os.path.basename(styles_df.loc[styles_df['id'] == int(rec[0]), 'image_path'].values[0])) for rec in similarities[:5]]
    return top_indices

def encode_text(text):
    text_tokens = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)
    return text_features

@app.route('/images/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['IMAGES_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)