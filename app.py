import os
import logging
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Mencegah matplotlib membuka window di server
import matplotlib.pyplot as plt
import math
from typing import List, Tuple
from werkzeug.utils import secure_filename
from joblib import load
from flask import Flask, render_template, send_from_directory, request, jsonify, redirect, url_for
import shutil

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Inisialisasi variabel dan style
plt.style.use('ggplot')

# Load model dan scaler di awal agar tidak perlu reload pada setiap request
best_svm_classifier = load('joblibs/fcc_svm.joblib')
try:
    y_pred = load('joblibs/y_pred_final.joblib')
except Exception as e:
    logger.warning(f'Could not load y_pred_final.joblib: {e}')
scaler = load('joblibs/scaler.joblib')

# Fungsi-fungsi utility
def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return (cnts, boundingBoxes)

def preprocess_image_for_prediction(image):
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return None
    
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    if image.shape[0] == 0 or image.shape[1] == 0:
        return None
    
    image = cv2.resize(image, (64, 64))
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel_bridge = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.uint8)
    bridged = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_bridge, iterations=1)
    
    try:
        if hasattr(cv2, 'ximgproc'):
            thinned = cv2.ximgproc.thinning(bridged, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        else:
            thinned = bridged
    except Exception as e:
        logger.debug(f'Thinning fallback: {e}')
        thinned = bridged
    
    if thinned is None or thinned.size == 0:
        return None
    
    scaled_image = thinned / 255.0
    return scaled_image

def extract_fcc_patch(patch: np.ndarray) -> List[int]:
    if np.max(patch) > 1:
        patch_binary = patch
    else:
        patch_binary = (patch * 255).astype(np.uint8)
    
    contours, _ = cv2.findContours(patch_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []
    
    largest_contour = max(contours, key=cv2.contourArea)
    if len(largest_contour) <= 2:
        return []
    
    chain_code = []
    for i in range(len(largest_contour) - 1):
        dx = largest_contour[i+1][0][0] - largest_contour[i][0][0]
        dy = largest_contour[i+1][0][1] - largest_contour[i][0][1]
        
        if dx == 1 and dy == 0: direction = 0
        elif dx == 1 and dy == -1: direction = 1
        elif dx == 0 and dy == -1: direction = 2
        elif dx == -1 and dy == -1: direction = 3
        elif dx == -1 and dy == 0: direction = 4
        elif dx == -1 and dy == 1: direction = 5
        elif dx == 0 and dy == 1: direction = 6
        elif dx == 1 and dy == 1: direction = 7
        else:
            if dx == 0 and dy == 0: continue
            angle = math.atan2(-dy, dx)
            angle = angle if angle >= 0 else angle + 2 * math.pi
            direction = int(round(angle / (math.pi / 4))) % 8
        chain_code.append(direction)
    return chain_code

def fcc_histogram(chain_code: List[int]) -> List[float]:
    if not chain_code:
        return [0] * 8
    hist = [0] * 8
    for direction in chain_code:
        if 0 <= direction < 8:
            hist[direction] += 1
    total = sum(hist)
    if total > 0:
        hist = [count / total for count in hist]
    return hist

def extract_fcc_features_from_image(img: np.ndarray, patch_size: Tuple[int, int] = (7, 7), stride: int = None) -> np.ndarray:
    h, w = img.shape
    grid_h, grid_w = 7, 7
    patch_height, patch_width = h // grid_h, w // grid_w
    features = []
    for i in range(grid_h):
        for j in range(grid_w):
            y_start = i * patch_height
            y_end = (i + 1) * patch_height if i < grid_h - 1 else h
            x_start = j * patch_width
            x_end = (j + 1) * patch_width if j < grid_w - 1 else w
            patch = img[y_start:y_end, x_start:x_end]
            fcc = extract_fcc_patch(patch)
            hist = fcc_histogram(fcc)
            features.extend(hist)
    return np.array(features)

def calculate_expected_fcc_feature_size(image_size, patch_size=(7, 7), stride=None):
    return 392  # 7x7 grid * 8 directions

def extract_fcc_features(image, patch_size=(7, 7), stride=None, visualize=True):
    if image is None or not isinstance(image, np.ndarray):
        return np.zeros((1, 392))
    
    image_uint8 = (image * 255).astype(np.uint8) if image.dtype != np.uint8 else image
    fcc_features = extract_fcc_features_from_image(image_uint8, patch_size=patch_size, stride=stride)
    
    if fcc_features is None or fcc_features.size == 0:
        fcc_features = np.zeros(392)
    
    features = fcc_features.flatten()
    expected_feature_size = calculate_expected_fcc_feature_size(image.shape, patch_size, stride)
    
    if len(features) < expected_feature_size:
        padded_features = np.zeros(expected_feature_size)
        padded_features[:len(features)] = features
        features = padded_features
    elif len(features) > expected_feature_size:
        features = features[:expected_feature_size]
    
    features = np.array(features)
    if features.ndim == 1:
        features = features.reshape(1, -1)
    return features

def recognize_single_character(image, expected_feature_size):
    preprocessed_image = preprocess_image_for_prediction(image)
    fcc_features = extract_fcc_features(preprocessed_image)
    
    if fcc_features.ndim == 1:
        fcc_features = fcc_features.reshape(1, -1)
    
    actual_feature_size = fcc_features.shape[1]
    if actual_feature_size != expected_feature_size:
        if actual_feature_size < expected_feature_size:
            padded_features = np.zeros((1, expected_feature_size))
            padded_features[0, :actual_feature_size] = fcc_features[0]
            fcc_features = padded_features
        else:
            fcc_features = fcc_features[:, :expected_feature_size]
    
    scaled_features = scaler.transform(fcc_features)
    
    try:
        raw_prediction = best_svm_classifier.predict(scaled_features)
        character = str(raw_prediction[0])
        return character, preprocessed_image
    except Exception as e:
        print(f"Exception details: {type(e).__name__}: {e}")
        return "unknown", preprocessed_image

def get_letters(img, filename, show_fig=False):
    letters = []
    inside_pisah = False
    pisah_buffer = []
    
    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    kernel = np.ones((30, 30), np.uint8)
    dilated = cv2.dilate(thresh1, kernel, iterations=1)
    cnts, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sort_contours(cnts, method="left-to-right")[0]
    
    dir_name = os.path.splitext(filename)[0]
    if not os.path.exists(f'history/{dir_name}/'):
        os.makedirs(f'history/{dir_name}/')
        
    for i, c in enumerate(cnts, start=1):
        if cv2.contourArea(c) > 20:
            x, y, w, h = cv2.boundingRect(c)
            roi = image[y:y + h, x:x + w]
            if roi is None or roi.size == 0 or roi.shape[0] == 0 or roi.shape[1] == 0:
                continue
                
            try:
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) > 2 else roi
                roi_64 = cv2.resize(roi_gray, (64, 64), interpolation=cv2.INTER_CUBIC)
                
                expected_feature_size = 392
                predicted_character, preprocessed_image = recognize_single_character(roi_64, expected_feature_size)
                
                if predicted_character is None or predicted_character == "unknown":
                    predicted_character = "?"
                
                pisah_buffer = predicted_character
                if len(pisah_buffer) >= 5 and pisah_buffer == 'pisah':
                    if not inside_pisah:
                        inside_pisah = True
                    else:
                        inside_pisah = False
                
                if inside_pisah:
                    if pisah_buffer == 'ga': predicted_character = "1"
                    if pisah_buffer == 'la': predicted_character = "7"
                    if pisah_buffer == 'ya': predicted_character = "9"
                
                preprocessed_display = (preprocessed_image * 255).astype(np.uint8)
                plt.figure(figsize=(12, 4))
                plt.subplot(1, 3, 1)
                plt.imshow(roi_gray, cmap='gray')
                plt.title(f'ROI Original ke-{i}')
                plt.axis('off')
                plt.subplot(1, 3, 2)
                plt.imshow(preprocessed_display, cmap='gray')
                plt.title('Preprocessing')
                plt.axis('off')
                plt.subplot(1, 3, 3)
                plt.imshow(preprocessed_display, cmap='gray')
                plt.title(f'Hasil Deteksi: {predicted_character}')
                plt.axis('off')
                plt.savefig(f'history/{dir_name}/{i}_bbox_{predicted_character}.png', bbox_inches='tight')
                plt.close()
                
                if predicted_character is not None:
                    letters.append(predicted_character)
                
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(image, predicted_character, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            except Exception as e:
                logger.warning(f'Error processing contour {i}: {e}')
                continue
                
    return letters, image

def recognize_and_display_image(image_path, filename, show_fig=False):
    image = cv2.imread(image_path)
    if image is None: return None, None
    try:
        letters, image_with_boxes = get_letters(image_path, filename, show_fig)
        dir_name = os.path.splitext(filename)[0]
        if not os.path.exists(f'history/{dir_name}/'):
            os.makedirs(f'history/{dir_name}/')
            
        plt.figure(figsize=(10, 6))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(f'Asli: {os.path.basename(image_path)}')
        plt.axis('off')
        plt.savefig(f'history/{dir_name}/original.png', bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(10, 6))
        plt.imshow(cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB))
        detected_text = "".join(letters)
        plt.title(f'Hasil Deteksi: {detected_text}')
        plt.axis('off')
        plt.savefig(f'history/{dir_name}/result.png', bbox_inches='tight')
        plt.close()
        
        return letters, image_with_boxes
    except Exception as e:
        logger.error(f'Error recognizing image {image_path}: {e}')
        return None, None

def predict_image(image_path, filename):
    letters, image_with_boxes = recognize_and_display_image(image_path, filename, show_fig=False)
    if letters is None or image_with_boxes is None: return []
    
    bounding_boxes = []
    dir_name = os.path.splitext(filename)[0]
    history_dir = f'history/{dir_name}'
    if not os.path.exists(history_dir): return []
    
    try:
        for bounding_box in sorted(os.listdir(history_dir)):
            if bounding_box not in ['result.png', 'result.gif', 'original.png']:
                full_path = os.path.join('history', dir_name, bounding_box)
                bounding_boxes.append(full_path)
        return bounding_boxes
    except Exception as e:
        logger.error(f'Error listing bounding boxes: {e}')
        return []

# INISIALISASI FLASK APP
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-ganti-saat-deploy')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max 16MB upload
UPLOAD_FOLDER = 'history'
SAMPLE_FOLDER = 'sample'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SAMPLE_FOLDER'] = SAMPLE_FOLDER

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        bounding_boxes = predict_image(image_path=os.path.join(app.config['UPLOAD_FOLDER'], filename), filename=filename)
        dir_name = os.path.splitext(filename)[0]
        result = f'/history/{dir_name}/result.png'
        return render_template('index.html', uploaded_filename=filename, bounding_boxes=bounding_boxes, result=result)
    return "Invalid file", 400

@app.route('/api/samples')
def get_samples():
    """Return list of sample image filenames."""
    sample_dir = app.config['SAMPLE_FOLDER']
    if not os.path.exists(sample_dir):
        return jsonify([])
    allowed_ext = {'png', 'jpg', 'jpeg', 'gif'}
    samples = [
        f for f in sorted(os.listdir(sample_dir))
        if '.' in f and f.rsplit('.', 1)[1].lower() in allowed_ext
    ]
    return jsonify(samples)

@app.route('/sample/<filename>')
def serve_sample(filename):
    """Serve a sample image file."""
    return send_from_directory(app.config['SAMPLE_FOLDER'], filename)

@app.route('/use-sample', methods=['POST'])
def use_sample():
    """Use a sample image for prediction."""
    sample_name = request.form.get('sample_name')
    if not sample_name:
        return "No sample selected", 400
    
    sample_path = os.path.join(app.config['SAMPLE_FOLDER'], sample_name)
    if not os.path.exists(sample_path):
        return "Sample not found", 404
    
    filename = secure_filename(sample_name)
    dest_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    shutil.copy2(sample_path, dest_path)
    
    bounding_boxes = predict_image(image_path=dest_path, filename=filename)
    dir_name = os.path.splitext(filename)[0]
    result = f'/history/{dir_name}/result.png'
    return render_template('index.html', uploaded_filename=filename, bounding_boxes=bounding_boxes, result=result, active_sample=sample_name)

@app.route('/history/<filename>')
def serve_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/history/<dir_name>/<filename>')
def serve_result(dir_name, filename):
    return send_from_directory(os.path.join(app.config['UPLOAD_FOLDER'], dir_name), filename)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)
