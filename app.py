import os
import json
import numpy as np
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, MobileNetV2
from sklearn.metrics.pairwise import cosine_similarity
import requests
import zipfile

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_DIR = "model"
FEATURES_FILE = os.path.join(MODEL_DIR, "image_features.json")
GITHUB_RELEASE_ZIP = "https://github.com/yourusername/yourrepo/releases/download/v1.0/model.zip"

def ensure_features_file():
    if not os.path.exists(FEATURES_FILE):
        print("⚠️ image_features.json not found. Downloading from GitHub release...")
        os.makedirs(MODEL_DIR, exist_ok=True)

        zip_path = os.path.join(MODEL_DIR, "model.zip")
        with requests.get(GITHUB_RELEASE_ZIP, stream=True) as r:
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(MODEL_DIR)

        os.remove(zip_path)
        print("✅ Download & extraction complete!")


ensure_features_file()

# Load model
model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

# Load saved image features
with open('model/image_features.json', 'r') as f:
    raw_features = json.load(f)

fashion_features = {
    fname: np.array(feat) / np.linalg.norm(feat)
    for fname, feat in raw_features.items()
}

def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)[0]
    return features / np.linalg.norm(features)

def recommend_styles(query_feature):
    similarities = {
        img_name: cosine_similarity([query_feature], [feat])[0][0]
        for img_name, feat in fashion_features.items()
    }
    top_images = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:5]
    return [{"filename": name, "score": round(score, 2)} for name, score in top_images]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/selfie', methods=['GET', 'POST'])
def selfie_page():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            return render_template('selfie.html', error="No file selected")

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        query_feature = extract_features(filepath)
        recommended_images = recommend_styles(query_feature)

        return render_template(
            'selfie.html',
            selfie_path=filepath,
            recommended_images=recommended_images
        )

    return render_template('selfie.html')

@app.route('/upload_selfie', methods=['POST'])
def upload_selfie():
    import base64
    from PIL import Image
    from io import BytesIO

    data_url = request.form['selfieData']
    header, encoded = data_url.split(",", 1)
    binary_data = base64.b64decode(encoded)

    selfie_path = os.path.join(UPLOAD_FOLDER, 'captured_selfie.png')

    try:
        img = Image.open(BytesIO(binary_data)).convert("RGB")
        img.save(selfie_path)
    except Exception as e:
        return render_template("selfie.html", error="Selfie capture failed.")

    query_feature = extract_features(selfie_path)
    recommended_images = recommend_styles(query_feature)

    return render_template(
        "selfie.html",
        selfie_path=selfie_path,
        recommended_images=recommended_images
    )

@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == '__main__':
    app.run(debug=True)







# # import os
# # import json
# # import numpy as np
# # from flask import Flask, render_template, request
# # from werkzeug.utils import secure_filename
# # from tensorflow.keras.preprocessing import image
# # from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, MobileNetV2
# # from sklearn.metrics.pairwise import cosine_similarity

# # app = Flask(__name__)
# # UPLOAD_FOLDER = 'static/uploads'
# # app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# # os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# # # Load model
# # model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

# # # Load saved image features
# # with open('model/image_features.json', 'r') as f:
# #     raw_features = json.load(f)

# # fashion_features = {
# #     fname: np.array(feat) / np.linalg.norm(feat)
# #     for fname, feat in raw_features.items()
# # }

# # def extract_features(img_path):
# #     img = image.load_img(img_path, target_size=(224, 224))
# #     img_array = image.img_to_array(img)
# #     img_array = np.expand_dims(img_array, axis=0)
# #     img_array = preprocess_input(img_array)
# #     features = model.predict(img_array)[0]
# #     return features / np.linalg.norm(features)

# # def recommend_styles(query_feature):
# #     similarities = {
# #         img_name: cosine_similarity([query_feature], [feat])[0][0]
# #         for img_name, feat in fashion_features.items()
# #     }
# #     top_images = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:5]
# #     return [{"filename": name, "score": round(score, 2)} for name, score in top_images]

# # @app.route('/')
# # def index():
# #     # Simple landing page or redirect
# #     return render_template('index.html')

# # @app.route('/selfie', methods=['GET', 'POST'])
# # def selfie_page():
# #     if request.method == 'POST':
# #         file = request.files.get('file')
# #         if not file or file.filename == '':
# #             return render_template('selfie.html', error="No file selected")

# #         filename = secure_filename(file.filename)
# #         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
# #         file.save(filepath)

# #         query_feature = extract_features(filepath)
# #         recommended_images = recommend_styles(query_feature)

# #         return render_template(
# #             'selfie.html',
# #             selfie_path=filepath,
# #             recommended_images=recommended_images
# #         )

# #     # GET → blank page
# #     return render_template('selfie.html')

# # @app.route('/upload_selfie', methods=['POST'])
# # def upload_selfie():
# #     import base64
# #     from PIL import Image
# #     from io import BytesIO

# #     data_url = request.form['selfieData']
# #     header, encoded = data_url.split(",", 1)
# #     binary_data = base64.b64decode(encoded)

# #     selfie_path = os.path.join(UPLOAD_FOLDER, 'captured_selfie.png')

# #     try:
# #         img = Image.open(BytesIO(binary_data)).convert("RGB")
# #         img.save(selfie_path)
# #     except Exception as e:
# #         return render_template("selfie.html", error="Selfie capture failed.")

# #     query_feature = extract_features(selfie_path)
# #     recommended_images = recommend_styles(query_feature)

# #     return render_template(
# #         "selfie.html",
# #         selfie_path=selfie_path,
# #         recommended_images=recommended_images
# #     )
    
    
# # if __name__ == '__main__':
# #     app.run(debug=True)

