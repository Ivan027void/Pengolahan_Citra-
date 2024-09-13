import os
from flask import Flask, render_template, request, redirect, url_for, jsonify
from matplotlib.figure import Figure
import cv2
from io import BytesIO
import base64
from utils import save_uploaded_file, get_image_url  # Import utility functions
from werkzeug.utils import secure_filename
import numpy as np

app = Flask(__name__)

# Set folder for uploaded images
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Define blur levels
#blur_levels = [5, 15, 25, 35, 45]

# Load OpenCV's pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Utility function to save the uploaded file
def save_uploaded_file(file, upload_folder):
    filename = secure_filename(file.filename)
    filepath = os.path.join(upload_folder, filename)
    file.save(filepath)
    return filename

# Utility function to get image URL for rendering
def get_image_url(filename):
    return f'/static/uploads/{filename}'

# Home route
@app.route('/')
def home():
    return render_template('home.html')

# About Us route
@app.route('/about')
def about():
    return render_template('about.html')

# Histogram Equalization route
@app.route('/Histogram_Equal', methods=['GET', 'POST'])
def histogram_equalization_page():
    if request.method == 'POST':
        if 'file' not in request.files or 'color_mode' not in request.form:
            return redirect(request.url)
        file = request.files['file']
        color_mode = request.form['color_mode']  # Grayscale or RGB
        if file.filename == '':
            return redirect(request.url)
        if file:
            # Save file using utility function
            filename = save_uploaded_file(file, app.config['UPLOAD_FOLDER'])
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # Read image
            if color_mode == 'grayscale':
                image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            else:
                image = cv2.imread(filepath)

            # Plot original histogram
            original_hist = plot_histogram(image, 'Original Histogram')

            # Perform histogram equalization
            equalized_image = histogram_equalization(image) if color_mode == 'rgb' else cv2.equalizeHist(image)
            equalized_hist = plot_histogram(equalized_image, 'Equalized Histogram')

            # Convert equalized image to base64 to display on HTML
            _, buffer = cv2.imencode('.png', equalized_image)
            equalized_image_data = base64.b64encode(buffer).decode('ascii')

            return render_template('histogram_equalization.html', 
                                   original_image=get_image_url(filename),  # Use utility function
                                   original_hist=original_hist,
                                   equalized_image=f"data:image/png;base64,{equalized_image_data}",
                                   equalized_hist=equalized_hist,
                                   color_mode=color_mode)
    return render_template('histogram_equalization.html')


# Face Detection route
@app.route('/Face_Detection', methods=['GET', 'POST'])
def face_detection_page():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            # Save the uploaded image file
            filename = save_uploaded_file(file, app.config['UPLOAD_FOLDER'])
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # Read the image
            image = cv2.imread(filepath)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detect faces using Haar Cascade
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

            # Draw rectangles around the faces and enhance accuracy by tweaking parameters
            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Count the number of detected faces
            face_count = len(faces)

            # Convert the processed image with rectangles to base64 for rendering in HTML
            _, buffer = cv2.imencode('.png', image)
            face_detected_image_data = base64.b64encode(buffer).decode('ascii')

            # Render the template with the original image, processed image, and face count
            return render_template('face_detection.html',
                                   original_image=get_image_url(filename),
                                   face_detected_image=f"data:image/png;base64,{face_detected_image_data}",
                                   face_count=face_count)

    # Render the face_detection.html template for GET requests
    return render_template('face_detection.html')


# Face Blurring route
@app.route('/Face_Blurring', methods=['GET', 'POST'])
def face_blurring_page():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            # Save file using utility function
            filename = save_uploaded_file(file, app.config['UPLOAD_FOLDER'])

            # Normalize the path for URL (use forward slashes)
            original_image_path = get_image_url(filename)  # Use utility function
            
            # Get the absolute path to the saved image
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Read the image using OpenCV
            img = cv2.imread(image_path)

            # Convert to grayscale for face detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Load OpenCV's Haar cascade for face detection
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            for (x, y, w, h) in faces:
                face = img[y:y+h, x:x+w]
                blurred_face = cv2.GaussianBlur(face, (15, 15), 0)
                img[y:y+h, x:x+w] = blurred_face
            
            _, buffer = cv2.imencode('.png', img)
            blurred_image_data = base64.b64encode(buffer).decode('ascii')
    
            # Render the initial page with the original image
            return render_template('face_blurring.html', 
                                   original_image=original_image_path,
                                   blurred_image= f"data:image/png;base64,{blurred_image_data}",
                                   blur_level=15)  # Default blur level
    return render_template('face_blurring.html')

# AJAX route to update blur effect
@app.route('/update_blur', methods=['POST'])
def update_blur():
    # Retrieve image path and blur level from the request
    filepath = request.json.get('filepath').replace('/static/', '')  # Correct the path to load from disk
    blur_level = int(request.json.get('blur_level'))
    if blur_level % 2 == 0:
        blur_level +=1

    # Read image (can handle both color and grayscale)
    full_filepath = os.path.join('static', filepath)  # Full path to image
    image = cv2.imread(full_filepath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Blur faces
    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        blurred_face = cv2.GaussianBlur(face, (blur_level, blur_level), 0)
        image[y:y+h, x:x+w] = blurred_face

    # Convert blurred image to base64 to send back to the client
    _, buffer = cv2.imencode('.png', image)
    blurred_image_data = base64.b64encode(buffer).decode('ascii')

    return jsonify({'blurred_image': f"data:image/png;base64,{blurred_image_data}"})

# Edge Detection route
@app.route('/Edge_Detection', methods=['GET', 'POST'])
def edge_detection_page():
    if request.method == 'POST':
        if 'file' not in request.files or 'detection_method' not in request.form:
            return redirect(request.url)
        file = request.files['file']
        detection_method = request.form['detection_method']  # Sobel, Canny, Prewitt, or Laplacian
        if file.filename == '':
            return redirect(request.url)
        if file:
            # Save file using utility function
            filename = save_uploaded_file(file, app.config['UPLOAD_FOLDER'])
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # Read image
            image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

            # Apply the selected edge detection method
            if detection_method == 'sobel':
                edges = apply_sobel(image)
            elif detection_method == 'canny':
                edges = cv2.Canny(image, 100, 200)
            elif detection_method == 'prewitt':
                edges = apply_prewitt(image)
            elif detection_method == 'laplacian':
                edges = cv2.Laplacian(image, cv2.CV_64F)
                edges = cv2.convertScaleAbs(edges)  # Convert back to uint8

            # Convert edge-detected image to base64 to display on HTML
            _, buffer = cv2.imencode('.png', edges)
            edge_image_data = base64.b64encode(buffer).decode('ascii')

            return render_template('edge_detection.html',
                                   original_image=get_image_url(filename),
                                   edge_detected_image=f"data:image/png;base64,{edge_image_data}",
                                   detection_method=detection_method)
    return render_template('edge_detection.html')

# Utility function for Sobel edge detection
def apply_sobel(image):
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    grad_x = cv2.convertScaleAbs(grad_x)
    grad_y = cv2.convertScaleAbs(grad_y)
    sobel = cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)
    return sobel

# # Utility function for Prewitt edge detection
# def apply_prewitt(image):
#     kernelx = cv2.getDerivKernels(1, 0, 3, normalize=True)[0]
#     kernely = cv2.getDerivKernels(0, 1, 3, normalize=True)[0]
#     prewitt_x = cv2.filter2D(image, -1, kernelx)
#     prewitt_y = cv2.filter2D(image, -1, kernely)
#     prewitt = cv2.addWeighted(prewitt_x, 0.5, prewitt_y, 0.5, 0)
#     return prewitt

def apply_prewitt(image):
    image = np.float32(image)
    # Define Prewitt kernels for horizontal and vertical edges
    kernel_x = np.array([[1, 0, -1],
                         [1, 0, -1],
                         [1, 0, -1]], dtype=np.float32)

    kernel_y = np.array([[1, 1, 1],
                         [0, 0, 0],
                         [-1, -1, -1]], dtype=np.float32)

    # Apply the kernels to the image using cv2.filter2D
    grad_x = cv2.filter2D(image, -1, kernel_x)
    grad_y = cv2.filter2D(image, -1, kernel_y)

    # Combine the gradients
    prewitt = cv2.sqrt(cv2.addWeighted(np.square(grad_x), 0.5, np.square(grad_y), 0.5, 0))
    prewitt = cv2.convertScaleAbs(prewitt)
    return prewitt


# Function to plot histogram
def plot_histogram(image, title):
    fig = Figure(figsize=(5, 3))
    ax = fig.add_subplot(111)
    
    if len(image.shape) == 2:  # Grayscale image
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        ax.plot(hist, color='gray')
    else:  # RGB image
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

# Function for histogram equalization
def histogram_equalization(image):
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img_output

if __name__ == "__main__":
    app.run(debug=True)