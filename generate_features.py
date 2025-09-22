# # import os
# import json
# import numpy as np
# from flask import Flask, request, render_template
# from werkzeug.utils import secure_filename
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, MobileNetV2
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.models import Model
# from sklearn.metrics.pairwise import cosine_similarity

# # Setup Flask
# app = Flask(__name__)
# UPLOAD_FOLDER = 'static/uploads'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # Load feature extractor
# base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
# model = Model(inputs=base_model.input, outputs=base_model.output)

# # Load precomputed fashion image features
# with open('model/image_features.json', 'r') as f:
#     fashion_features = json.load(f)

# # Function to extract features from an uploaded selfie
# def extract_features(img_path):
#     img = image.load_img(img_path, target_size=(224, 224))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = preprocess_input(img_array)
#     features = model.predict(img_array)
#     return features.flatten()

# # Recommend top 5 similar fashion styles
# def recommend_styles(query_feature):
#     similarities = {}
#     for img_name, feat in fashion_features.items():
#         sim = cosine_similarity([query_feature], [feat])[0][0]
#         similarities[img_name] = sim
#     top_images = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:5]
#     return [img[0] for img in top_images]

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         # Save uploaded selfie
#         file = request.files['file']
#         if file.filename == '':
#             return render_template('index.html', error="No file selected")
#         filename = secure_filename(file.filename)
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(filepath)

#         # Extract features and recommend styles
#         query_feature = extract_features(filepath)
#         recommended_images = recommend_styles(query_feature)

#         return render_template('index.html',
#                                selfie_path=filepath,
#                                recommended_images=recommended_images)

#     return render_template('index.html')

# if __name__ == '__main__':
#     app.run(debug=True)
import os
import json
import numpy as np
import pandas as pd
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image

# Load gender labels
df = pd.read_csv("styles.csv")  # Assumes columns: filename, gender
gender_dict = dict(zip(df['filename'], df['gender']))

# Load model
model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

# Paths
IMAGE_FOLDER = 'static/fashion_images'
OUTPUT_JSON = 'model/image_features.json'
os.makedirs("model", exist_ok=True)

# Extract features
features = {}

for fname in os.listdir(IMAGE_FOLDER):
    if fname.lower().endswith(('.jpg', '.jpeg', '.png')) and fname in gender_dict:
        img_path = os.path.join(IMAGE_FOLDER, fname)
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = preprocess_input(np.expand_dims(x, axis=0))
        feat = model.predict(x)[0].tolist()

        features[fname] = {
            "feature": feat,
            "gender": gender_dict[fname].strip().lower()
        }

with open(OUTPUT_JSON, "w") as f:
    json.dump(features, f, indent=2)

print("✅ Features with gender saved.")
