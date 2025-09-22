import os
import json
import numpy as np
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import preprocess_input, MobileNetV2
from sklearn.metrics.pairwise import cosine_similarity
import requests
import zipfile
import io
import glob
from tqdm import tqdm

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

FEATURES_ZIP_URL = "https://github.com/Mahnoor-Sadiq/AI-Powered-Recommended-By-Selfie/releases/download/ML/image_features.zip"
IMAGES_ZIP_URL = "https://github.com/Mahnoor-Sadiq/AI-Powered-Recommended-By-Selfie/releases/download/ML/fashion_images.zip"

FEATURES_PATH = "model/image_features.json"
IMAGES_DIR = "static/fashion_images/"


def download_and_extract_zip(url, extract_to, max_retries=5, chunk_size=1024*1024):
    os.makedirs(extract_to, exist_ok=True)
    local_zip = os.path.join(extract_to, "temp_download.zip")

    # Retry logic
    for attempt in range(1, max_retries + 1):
        try:
            # Support resuming
            resume_header = {}
            pos = 0
            if os.path.exists(local_zip):
                pos = os.path.getsize(local_zip)
                resume_header = {"Range": f"bytes={pos}-"}
                print(f"Resuming download from {pos/1024/1024:.2f} MB...")

            with requests.get(url, headers=resume_header, stream=True, timeout=30) as r:
                r.raise_for_status()
                total_size = int(r.headers.get("Content-Length", 0)) + pos
                progress = tqdm(
                    total=total_size,
                    initial=pos,
                    unit="B",
                    unit_scale=True,
                    desc=f"Downloading"
                )

                with open(local_zip, "ab") as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            progress.update(len(chunk))
                progress.close()

            print("Download complete. Extracting...")
            with zipfile.ZipFile(local_zip, "r") as zip_ref:
                zip_ref.extractall(extract_to)

            os.remove(local_zip)
            print("Extraction done ✅")
            return  # success, exit function

        except Exception as e:
            print(f"Attempt {attempt} failed: {e}")
            if attempt == max_retries:
                raise
            else:
                print("Retrying download...")

def ensure_files():
    # 1. Ensure model/image_features.json
    if not os.path.exists(FEATURES_PATH):
        print("image_features.json not found. Downloading...")
        download_and_extract_zip(FEATURES_ZIP_URL, "model")
    else:
        print("image_features.json already exists ✅")

    # 2. Ensure static/fashion_images/ has images
    if not os.path.exists(IMAGES_DIR) or not glob.glob(os.path.join(IMAGES_DIR, "*")):
        print("Fashion images not found. Downloading...")
        download_and_extract_zip(IMAGES_ZIP_URL, "static")
    else:
        print("Fashion images already exist ✅")

ensure_files()

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

