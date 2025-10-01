from flask import Flask, request, send_file, jsonify, render_template
from flask_cors import CORS
import cv2
import numpy as np
import tempfile
import os
from io import BytesIO
import base64
import json

app = Flask(__name__)
CORS(app)

# ==== Thuật toán xử lý ảnh ====
def blur_image(img, method="blur", ksize=5, sigma=1):
    ksize = max(1, ksize)
    if ksize % 2 == 0:
        ksize += 1
        
    if method == "blur":
        return cv2.blur(img, (ksize, ksize))
    elif method == "gaussian":
        return cv2.GaussianBlur(img, (ksize, ksize), sigma)
    elif method == "median":
        return cv2.medianBlur(img, ksize)
    else:
        raise ValueError("Unknown blur method!")

def sharpen_image(img):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)

def edge_detection(img, method="sobel", threshold1=100, threshold2=200):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if method == "sobel":
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edges = cv2.convertScaleAbs(cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0))
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    elif method == "laplacian":
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        edges = cv2.convertScaleAbs(laplacian)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    elif method == "canny":
        edges = cv2.Canny(gray, threshold1, threshold2)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    else:
        raise ValueError("Unknown edge detection method!")

def calculate_histogram(img):
    """Tính histogram cho từng kênh màu"""
    histograms = []
    colors = ['b', 'g', 'r']
    
    for i, color in enumerate(colors):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        histograms.append(hist.flatten().tolist())
    
    return {
        'blue': histograms[0],
        'green': histograms[1],
        'red': histograms[2]
    }

# ==== Routes ====
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process():
    try:
        if "file" not in request.files:
            return jsonify({"error": "Không có file được tải lên"}), 400
            
        file = request.files["file"]
        if file.filename == '':
            return jsonify({"error": "Không có file được chọn"}), 400
            
        # Lấy parameters từ form
        method = request.form.get("method", "blur")
        ksize = int(request.form.get("ksize", 5))
        sigma = float(request.form.get("sigma", 1))
        th1 = int(request.form.get("th1", 100))
        th2 = int(request.form.get("th2", 200))
        
        # Đọc ảnh từ upload
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"error": "Không thể đọc file ảnh"}), 400
        
        # Tính histogram ảnh gốc
        original_hist = calculate_histogram(img)
        
        # Xử lý ảnh theo method được chọn
        if method in ["blur", "gaussian", "median"]:
            result = blur_image(img, method, ksize, sigma)
        elif method == "sharpen":
            result = sharpen_image(img)
        elif method in ["sobel", "laplacian", "canny"]:
            result = edge_detection(img, method, th1, th2)
        else:
            return jsonify({"error": "Phương thức không hợp lệ"}), 400
        
        # Tính histogram ảnh đã xử lý
        processed_hist = calculate_histogram(result)
        
        # Encode ảnh gốc và ảnh đã xử lý thành base64
        _, original_buffer = cv2.imencode('.jpg', img)
        original_base64 = base64.b64encode(original_buffer).decode('utf-8')
        
        _, processed_buffer = cv2.imencode('.jpg', result)
        processed_base64 = base64.b64encode(processed_buffer).decode('utf-8')
        
        # Trả về JSON chứa cả ảnh và histogram
        return jsonify({
            "original_image": f"data:image/jpeg;base64,{original_base64}",
            "processed_image": f"data:image/jpeg;base64,{processed_base64}",
            "original_histogram": original_hist,
            "processed_histogram": processed_hist
        })
            
    except Exception as e:
        return jsonify({"error": f"Lỗi xử lý: {str(e)}"}), 500

@app.route("/preview", methods=["POST"])
def preview():
    """API để preview ảnh gốc"""
    try:
        if "file" not in request.files:
            return jsonify({"error": "Không có file được tải lên"}), 400
            
        file = request.files["file"]
        if file.filename == '':
            return jsonify({"error": "Không có file được chọn"}), 400
            
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"error": "Không thể đọc file ảnh"}), 400
        
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            cv2.imwrite(tmp.name, img)
            return send_file(tmp.name, mimetype="image/jpeg", as_attachment=False)
            
    except Exception as e:
        return jsonify({"error": f"Lỗi preview: {str(e)}"}), 500

if __name__ == "__main__":
    os.makedirs("templates", exist_ok=True)
    print("🚀 CaiTiemPhoto Server đang chạy tại: http://localhost:5000")
    print("📸 Mở trình duyệt và truy cập địa chỉ trên để sử dụng!")
    app.run(host="127.0.0.1", port=5000, debug=True)