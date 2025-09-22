# AI-Powered Outfit Recommendation by Selfie

This project uses **TensorFlow MobileNetV2** to extract features from a selfie and recommend similar fashion styles from a dataset of images.  

On the first run, the app will **automatically download required assets** (fashion images and precomputed image features JSON) from GitHub Releases. These are large files, so it may take some time depending on your internet connection. Once downloaded, they are cached locally, and subsequent runs will start instantly.

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Create and activate a virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the project
```bash
python app.py
```

---

## 📥 First-Time Setup (Automatic Download)

- On the very first run, the app will:
  - Check if `model/image_features.json` exists.  
    - If not, it will **download `image_features.zip`** and extract it to the `model/` folder.  
  - Check if `static/fashion_images/` contains images.  
    - If not, it will **download `fashion_images.zip`** and extract it to `static/fashion_images/`.  

After this initial download, the assets are stored locally and the app will not need to fetch them again.

If the download is interrupted, simply restart the app — it supports **resume and retries**.

---

## 🌐 Usage

- Open your browser and go to:
  ```
  http://127.0.0.1:5000/
  ```
- Upload or capture a selfie.  
- The system will recommend the top 5 fashion images that are visually similar.  

---

## ⚠️ Notes

- The first download may be **hundreds of MBs**, so make sure you have a stable internet connection.  
- Requires **Python 3.8+**.  
- TensorFlow will run on CPU by default.  
  - If you have a compatible GPU, you can install `tensorflow-gpu` for faster performance.  

---

## 📄 Author
Mahnoor Sadiq

