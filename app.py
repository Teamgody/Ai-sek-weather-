from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
import numpy as np
import base64
import io
import cv2
import os
from skimage.metrics import structural_similarity as ssim

app = Flask(__name__)
CORS(app)

def is_sky_image(image_array):
    hsv = cv2.cvtColor(image_array, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    blue_ratio = np.sum(mask_blue > 0) / (image_array.shape[0] * image_array.shape[1])

    lower_gray = np.array([0, 0, 80])
    upper_gray = np.array([180, 50, 200])
    mask_gray = cv2.inRange(hsv, lower_gray, upper_gray)
    gray_ratio = np.sum(mask_gray > 0) / (image_array.shape[0] * image_array.shape[1])

    brightness = np.mean(cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY))
    return (blue_ratio > 0.15 or gray_ratio > 0.25) and brightness > 40

def estimate_time_of_day_from_image(avg_brightness):
    if avg_brightness < 70:
        return "ช่วงกลางคืน"
    elif avg_brightness < 130:
        return "ช่วงเย็น"
    elif avg_brightness < 200:
        return "ช่วงบ่าย"
    else:
        return "ช่วงเช้า"

def analyze_cloud_and_rain(image_array):
    image = cv2.resize(image_array, (300, 300))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_cloud = np.array([0, 0, 180])
    upper_cloud = np.array([180, 50, 255])
    cloud_mask = cv2.inRange(hsv, lower_cloud, upper_cloud)
    cloud_pixels = cv2.countNonZero(cloud_mask)
    total_pixels = image.shape[0] * image.shape[1]
    cloud_percent = (cloud_pixels / total_pixels) * 100
    if cloud_percent > 30:
        rain_chance = 50
    elif cloud_percent > 20:
        rain_chance = 40
    elif cloud_percent > 15:
        rain_chance = 30
    elif cloud_percent > 10:
        rain_chance = 10
    else:
        rain_chance = 5
    return round(cloud_percent, 2), rain_chance

def compare_with_reference(image_cv2, reference_folder='static/reference_sky'):
    max_score = 0
    best_match = None

    for filename in os.listdir(reference_folder):
        ref_path = os.path.join(reference_folder, filename)
        ref_img = cv2.imread(ref_path)
        if ref_img is None:
            continue
        resized = cv2.resize(image_cv2, (ref_img.shape[1], ref_img.shape[0]))
        gray1 = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
        score, _ = ssim(gray1, gray2, full=True)
        if score > max_score:
            max_score = score
            best_match = filename

    return best_match, max_score

@app.route('/', methods=['GET', 'HEAD'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'result': 'No image uploaded'}), 400

    file = request.files['image']
    image_pil = Image.open(file.stream).convert('RGB')
    image_array = np.array(image_pil)
    image_cv2 = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

    if not is_sky_image(image_cv2):
        return jsonify({'result': 'ไม่ใช่ภาพท้องฟ้า', 'image': None})

    avg_brightness = np.mean(image_array)
    if avg_brightness < 100:
        weather = "ฝนตกหรือครึ้มฟ้า"
    elif avg_brightness < 180:
        weather = "อากาศปกติ"
    else:
        weather = "แดดจัดหรือท้องฟ้าโปร่ง"

    time_period = estimate_time_of_day_from_image(avg_brightness)
    cloud_percent, rain_chance = analyze_cloud_and_rain(image_cv2)

    best_match, score = compare_with_reference(image_cv2)
    if score > 0.45:
        label = best_match.replace('.jpg', '').replace('_', ' ')
        similarity_text = f"คล้ายกับภาพตัวอย่าง: {label} (ความเหมือน {round(score, 2)})"
    else:
        similarity_text = "ไม่มีภาพตัวอย่างที่คล้ายมากพอ"

    result = f"{weather} ({time_period})\\nเมฆปกคลุม: {cloud_percent}%\\nโอกาสฝนตก: {rain_chance}%\\n{similarity_text}"

    buffered = io.BytesIO()
    image_pil.save(buffered, format="PNG")
    encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return jsonify({'result': result, 'image': encoded_image})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
