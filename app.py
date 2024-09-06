from flask import Flask, render_template, request, redirect, url_for
from matplotlib.figure import Figure
import cv2
import os
from io import BytesIO
import base64

app = Flask(__name__)

UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Fungsi untuk membuat plot histogram
# Fungsi untuk membuat plot histogram
def plot_histogram(image, title):
    fig = Figure(figsize=(5, 3))
    ax = fig.add_subplot(111)
    
    if len(image.shape) == 2:  # Grayscale image
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        ax.plot(hist, color='gray')
    else:  # RGB image
        # Hitung histogram untuk setiap channel (B, G, R)
        for i, color in enumerate(['b', 'g', 'r']):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            ax.plot(hist, color=color)
    
    ax.set_title(title)
    ax.set_xlabel("Intensity Value")
    ax.set_ylabel("Frequency")
    ax.set_xlim([0, 256])
    fig.tight_layout()
        
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
        
    return f"data:image/png;base64,{data}"

    
# Fungsi untuk histogram equalization
def histogram_equalization(image):
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img_output

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files or 'color_mode' not in request.form:
            return redirect(request.url)
        file = request.files['file']
        color_mode = request.form['color_mode']  # Grayscale or RGB
        if file.filename == '':
            return redirect(request.url)
        if file:
            # Simpan file
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Baca gambar
            if color_mode == 'grayscale':
                image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            else:
                image = cv2.imread(filepath)

            # Plot histogram asli
            original_hist = plot_histogram(image, 'Original Histogram')

            # Histogram equalization
            equalized_image = histogram_equalization(image) if color_mode == 'rgb' else cv2.equalizeHist(image)
            equalized_hist = plot_histogram(equalized_image, 'Equalized Histogram')

            # Konversi gambar hasil equalization ke base64 untuk ditampilkan di HTML
            _, buffer = cv2.imencode('.png', equalized_image)
            equalized_image_data = base64.b64encode(buffer).decode('ascii')

            return render_template('index.html', 
                                   original_image=file.filename, 
                                   original_hist=original_hist,
                                   equalized_image=f"data:image/png;base64,{equalized_image_data}",
                                   equalized_hist=equalized_hist,
                                   color_mode=color_mode)
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
