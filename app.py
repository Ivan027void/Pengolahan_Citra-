# from flask import Flask, request, render_template, redirect, url_for
# import cv2
# import numpy as np
# import os
# from matplotlib import pyplot as plt
# from werkzeug.utils import secure_filename

# app = Flask(__name__)

# # Set up the upload folder
# UPLOAD_FOLDER = 'static/uploads'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # Function to save and return image path
# def save_image(image, filename):
#     path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#     cv2.imwrite(path, image)
#     return path

# # Function to create color histogram
# def create_color_histogram(image, hist_path):
#     plt.figure()
#     colors = ('b', 'g', 'r')
#     for i, color in enumerate(colors):
#         hist = cv2.calcHist([image], [i], None, [256], [0, 256])
#         plt.plot(hist, color=color)
#         plt.xlim([0, 256])
#     plt.title('Color Histogram')
#     plt.xlabel('Intensity Value')
#     plt.ylabel('Frequency')
#     plt.grid(True)
#     plt.savefig(hist_path)
#     plt.close()

# # Function to perform histogram equalization on a color image
# def histogram_equalization(image_path):
#     image = cv2.imread(image_path)
#     img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
#     img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
#     equalized_image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
#     return equalized_image

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         # Check if the post request has the file part
#         if 'file' not in request.files:
#             return redirect(request.url)
#         file = request.files['file']
#         # If the user does not select a file, the browser submits an empty file without a filename
#         if file.filename == '':
#             return redirect(request.url)
#         if file:
#             # Save the file
#             filename = secure_filename(file.filename)
#             image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#             file.save(image_path)

#             # Read the original image and create color histogram
#             original_image = cv2.imread(image_path)
#             original_hist_path = os.path.join(app.config['UPLOAD_FOLDER'], 'original_hist.png')
#             create_color_histogram(original_image, original_hist_path)

#             # Paths for equalized image and histogram
#             equalized_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'equalized.jpg')
#             equalized_hist_path = os.path.join(app.config['UPLOAD_FOLDER'], 'equalized_hist.png')

#             # Check if equalization button was clicked
#             if 'equalize' in request.form:
#                 equalized_image = histogram_equalization(image_path)
#                 save_image(equalized_image, 'equalized.jpg')

#                 # Create color histogram for equalized image
#                 create_color_histogram(equalized_image, equalized_hist_path)

#                 return render_template('index.html',
#                                        original_image=url_for('static', filename=f'uploads/{filename}'),
#                                        equalized_image=url_for('static', filename='uploads/equalized.jpg'),
#                                        original_hist=url_for('static', filename='uploads/original_hist.png'),
#                                        equalized_hist=url_for('static', filename='uploads/equalized_hist.png'))

#             return render_template('index.html', 
#                                    original_image=url_for('static', filename=f'uploads/{filename}'),
#                                    original_hist=url_for('static', filename='uploads/original_hist.png'))

#     return render_template('index.html')

# if __name__ == '__main__':
#     app.run(debug=True)
from flask import Flask, render_template, request, redirect, url_for
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from io import BytesIO
import base64

app = Flask(__name__)

UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Fungsi untuk membuat plot histogram
def plot_histogram(image, title):
    fig = Figure(figsize=(5, 3))
    ax = fig.add_subplot(111)
    

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
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            # Simpan file
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Baca gambar
            image = cv2.imread(filepath)

            # Plot histogram asli
            original_hist = plot_histogram(image, 'Original Histogram')

            # Histogram equalization
            equalized_image = histogram_equalization(image)
            equalized_hist = plot_histogram(equalized_image, 'Equalized Histogram')

            # Konversi gambar hasil equalization ke base64 untuk ditampilkan di HTML
            _, buffer = cv2.imencode('.png', equalized_image)
            equalized_image_data = base64.b64encode(buffer).decode('ascii')

            return render_template('index.html', 
                                   original_image=file.filename, 
                                   original_hist=original_hist,
                                   equalized_image=f"data:image/png;base64,{equalized_image_data}",
                                   equalized_hist=equalized_hist)
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
