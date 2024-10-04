import os
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file
from matplotlib.figure import Figure
import cv2
from io import BytesIO
import base64
from werkzeug.utils import secure_filename
import numpy as np
import random
from PIL import Image

app = Flask(__name__)

# Set folder for uploaded images
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load OpenCV's pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Utility function to save the uploaded file
def save_uploaded_file(file, upload_folder):
    filename = secure_filename(file.filename)
    filepath = os.path.join(upload_folder, filename)
    file.save(filepath)
    return filename

# Utility function to get image URL for rendering
def get_image_url(filename):
    return f'/static/uploads/{filename}'

# Convert images to base64 for rendering in HTML
def convert_image_to_base64(img):
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('ascii')

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
    prewitt = cv2.sqrt(cv2.addWeighted(np.square(grad_x), 1.0, np.square(grad_y), 1.0, 0))
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

# Segmentation route for satellite images
@app.route('/Segment_Image', methods=['GET', 'POST'])
def segment_image():
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

            # Read the satellite image
            image = cv2.imread(filepath)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # Define the green color range for vegetation
            lower_green = np.array([35, 50, 50])  # Lower range for green color (HSV)
            upper_green = np.array([90, 255, 255])  # Upper range for green color (HSV)

            # Create a mask for green (vegetation)
            mask = cv2.inRange(hsv, lower_green, upper_green)
            segmented_with_plants = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)

            # Region without vegetation (inverse of the plant mask)
            mask_inverse = cv2.bitwise_not(mask)
            segmented_without_plants = cv2.bitwise_and(image_rgb, image_rgb, mask=mask_inverse)
            # keep the rgb color on the image
            segmented_without_plants = cv2.cvtColor(segmented_without_plants, cv2.COLOR_BGR2RGB)

            plants_image_base64 = convert_image_to_base64(segmented_with_plants)
            non_plants_image_base64 = convert_image_to_base64(segmented_without_plants)

            # Render the results on the template
            return render_template('segmentation_image.html',
                                   original_image=get_image_url(filename),
                                   plants_image=f"data:image/png;base64,{plants_image_base64}",
                                   non_plants_image=f"data:image/png;base64,{non_plants_image_base64}")

    return render_template('segmentation_image.html')  # Render upload form for GET request

# Color Filtering route
@app.route('/Color_Filtering', methods=['GET', 'POST'])
def color_filtering_page():
    if request.method == 'POST':
        if 'file' not in request.files or 'color' not in request.form:
            return redirect(request.url)
        file = request.files['file']
        color = request.form['color']  # RGB or specific color like red, green, blue
        if file.filename == '':
            return redirect(request.url)
        if file:
            # Save file using utility function
            filename = save_uploaded_file(file, app.config['UPLOAD_FOLDER'])
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # Read image
            image = cv2.imread(filepath)
            image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # Define color ranges for filtering (adjust based on color)
            if color == 'red':
                lower_bound = np.array([0, 120, 70])
                upper_bound = np.array([10, 255, 255])
            elif color == 'green':
                lower_bound = np.array([35, 100, 100])
                upper_bound = np.array([85, 255, 255])
            elif color == 'blue':
                lower_bound = np.array([100, 150, 0])
                upper_bound = np.array([140, 255, 255])
            else:
                # Default is no filtering
                lower_bound = np.array([0, 0, 0])
                upper_bound = np.array([180, 255, 255])

            # Apply color mask
            mask = cv2.inRange(image_hsv, lower_bound, upper_bound)
            filtered_image = cv2.bitwise_and(image, image, mask=mask)

            # Convert the filtered image to base64 to display on HTML
            _, buffer = cv2.imencode('.png', filtered_image)
            filtered_image_data = base64.b64encode(buffer).decode('ascii')

            return render_template('color_filtering.html',
                                   original_image=get_image_url(filename),
                                   filtered_image=f"data:image/png;base64,{filtered_image_data}",
                                   color=color)

    return render_template('color_filtering.html')

# fitur konversi gambar ke efek filter kamera.
@app.route('/Image_Effects', methods=['GET', 'POST'])
def image_effects_page():
    if request.method == 'POST':
        if 'file' not in request.files or 'effect' not in request.form:
            return redirect(request.url)
        file = request.files['file']
        effect = request.form['effect']  # grayscale, sepia, negative, lomo
        if file.filename == '':
            return redirect(request.url)
        if file:
            # Save file using utility function
            filename = save_uploaded_file(file, app.config['UPLOAD_FOLDER'])
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # Read image
            image = cv2.imread(filepath)

            # Apply the selected effect
            if effect == 'grayscale':
                effect_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif effect == 'black_and_white':
                effect_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                _, effect_image = cv2.threshold(effect_image, 128, 255, cv2.THRESH_BINARY)
            elif effect == 'glitch':
                effect_image = apply_glitch(image)
            elif effect == 'pixel':
                effect_image = apply_pixel(image)
            elif effect == 'vaporwave':
                effect_image = apply_vaporwave(image)
            elif effect == 'duotone':
                effect_image = apply_duotone(image)
            elif effect == 'split_tone':
                effect_image = apply_split_tone(image)
            elif effect == 'negative':
                effect_image = 255 - image
            elif effect == 'retro':
                effect_image = apply_retro(image)
            elif effect == 'noir':
                effect_image = apply_noir(image)
        
            # Convert the effect image to base64 using func
            effect_image_data = convert_image_to_base64(effect_image)
            

            return render_template('image_effects.html',
                                   original_image=get_image_url(filename),
                                   effect_image=f"data:image/png;base64,{effect_image_data}",
                                   effect=effect)
    
    return render_template('image_effects.html')


# Utility function to apply glitch 
def apply_glitch(image):
    # Create a copy of the image to work with
    glitch_image = image.copy()

    # Step 1: Randomly shift horizontal slices of the image
    rows, cols, _ = image.shape
    slice_height = np.random.randint(5, 20)  # Random slice height between 5 and 20 pixels

    for i in range(0, rows, slice_height):
        # Define slice range
        start_row = i
        end_row = min(i + slice_height, rows)
        
        # Randomly shift the slice
        shift = np.random.randint(-10, 10)  # Random shift amount
        glitch_image[start_row:end_row] = np.roll(glitch_image[start_row:end_row], shift, axis=1)

    # Step 2: Color channel offset
    for channel in range(3):  # For B, G, R channels
        # Randomly shift each channel to create a color separation effect
        channel_shift = np.random.randint(-5, 5)
        # Shift the channel
        channel_data = glitch_image[:, :, channel]
        glitch_image[:, :, channel] = np.roll(channel_data, channel_shift, axis=1)

    # Step 3: Add noise
    noise = np.random.randint(0, 50, (rows, cols, 3), dtype=np.uint8)  # Create random noise
    glitch_image = cv2.add(glitch_image, noise)  # Add noise to the image

    # Clip values to ensure they remain valid
    glitch_image = np.clip(glitch_image, 0, 255).astype(np.uint8)

    return glitch_image

# Utility function to apply pixel effect
def apply_pixel(image, pixel_size=10):
    # Pixelate the image
    height, width, _ = image.shape
    pixelated = cv2.resize(image, (width // pixel_size, height // pixel_size), interpolation=cv2.INTER_NEAREST)
    pixelated_image = cv2.resize(pixelated, (width, height), interpolation=cv2.INTER_NEAREST)
    return pixelated_image

# Utility function to apply vaporwave effect
def apply_vaporwave(image):
    # Create a copy of the image to work with
    vaporwave_image = image.copy()

    # Step 1: Apply a color palette (pastel colors)
    # Convert the image to float for color manipulation
    vaporwave_image = vaporwave_image.astype(np.float32) / 255.0

    # Apply a color shift to create a pastel effect
    pastel_color_shift = np.array([0.8,0.5,1.0])  # RGB (1.0 -> R, 0.5 -> G, 0.8 -> B)
    vaporwave_image *= pastel_color_shift  # Multiply color channels
    vaporwave_image = np.clip(vaporwave_image, 0, 1)  # Ensure values are within [0, 1]

    # Step 2: Add a gradient overlay
    height, width, _ = vaporwave_image.shape
    gradient = np.linspace(0, 1, height).reshape(height, 1)  # Vertical gradient
    gradient = np.tile(gradient, (1, width))  # Tile to cover the width
    gradient = np.expand_dims(gradient, axis=-1)  # Add channel dimension

    # Apply the gradient as an overlay
    vaporwave_image += gradient * 0.2  # Adjust intensity
    vaporwave_image = np.clip(vaporwave_image, 0, 1)  # Ensure values are within [0, 1]

    # Step 3: Add a Gaussian blur
    vaporwave_image = cv2.GaussianBlur(vaporwave_image, (5, 5), 0)

    # Step 4: Add noise for texture
    noise = np.random.normal(0, 0.1, vaporwave_image.shape).astype(np.float32)  # Gaussian noise
    vaporwave_image += noise
    vaporwave_image = np.clip(vaporwave_image, 0, 1)  # Ensure values are within [0, 1]

    # Step 5: Convert back to uint8
    return (vaporwave_image * 255).astype(np.uint8)

# Utility function to apply duotone effect
def apply_duotone(image):
    # Replace the colors in the image to create a duotone effect
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Normalize the grayscale image
    norm_image = cv2.normalize(gray_image, None, 0, 255, cv2.NORM_MINMAX)
    
    # Create two colors for the duotone effect
    color1 = np.array([255, 0, 0])  # Red in BGR
    color2 = np.array([0, 0, 255])  # Blue in BGR
    
    # Map the normalized image to the two colors
    duotone_image = np.zeros_like(image)
    for i in range(3):  # For each channel
        duotone_image[:, :, i] = norm_image * (color1[i] * (norm_image <= 127) + color2[i] * (norm_image > 127)) / 255
    
    return duotone_image.astype(np.uint8)

# Utility function to apply split tone effect
def apply_split_tone(image, highlight_color=(0, 195, 255), shadow_color=(0, 100, 0)):
    # Convert to float for precise manipulation
    split_tone_image = image.astype(np.float32) / 255.0

    # Create an output image with the same shape as the input
    output_image = np.zeros_like(image, dtype=np.float32)

    # Create masks for highlights and shadows
    highlight_mask = split_tone_image.mean(axis=2) > 0.5  # Mean to find highlights
    shadow_mask = split_tone_image.mean(axis=2) <= 0.5  # Mean to find shadows

    # Apply highlight color
    output_image[highlight_mask] = highlight_color  # Direct assignment for highlights

    # Apply shadow color
    output_image[shadow_mask] = shadow_color  # Direct assignment for shadows

    # Convert back to uint8 and ensure values are clipped to the valid range
    output_image = np.clip(output_image, 0, 255).astype(np.uint8)

    return output_image

# Utility function to apply retro sepia effect
def apply_retro(image, sepia_intensity=1, contrast_reduction=0.7, noise_amount=0.02):
    """Applies a retro filter with adjustable sepia intensity and adds noise for an old photo look.
    
    Args:
        image: The input image in BGR format.
        sepia_intensity: Float to adjust sepia strength (1 for sepia, lower for vintage).
        contrast_reduction: Float to adjust contrast (lower for more washed-out retro look).
        noise_amount: Float to adjust the amount of noise (higher adds more grain).

    Returns:
        The image with the retro effect and noise applied.
    """
    # Sepia filter matrix
    sepia_filter = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])

    # Apply sepia filter with intensity adjustment
    sepia_image = cv2.transform(image, sepia_filter * sepia_intensity)
    sepia_image = np.clip(sepia_image, 0, 255).astype(np.uint8)

    # Reduce contrast slightly (for retro effect)
    contrast_reduced = cv2.addWeighted(sepia_image, contrast_reduction, np.zeros_like(sepia_image), 0, 30)

    # Add impulse noise (salt-and-pepper noise)
    noisy_image = add_impulse_noise(contrast_reduced, noise_amount)

    return noisy_image

def add_impulse_noise(image, noise_amount):
    """Adds salt-and-pepper noise to the image.
    
    Args:
        image: The input image in BGR format.
        noise_amount: Float, the proportion of pixels to be affected by noise.

    Returns:
        The noisy image.
    """
    noisy_image = image.copy()
    h, w, _ = noisy_image.shape
    num_noise_pixels = int(noise_amount * h * w)
    
    # Add 'salt' (white) noise
    for _ in range(num_noise_pixels // 2):
        y = random.randint(0, h - 1)
        x = random.randint(0, w - 1)
        noisy_image[y, x] = [255, 255, 255]

    # Add 'pepper' (black) noise
    for _ in range(num_noise_pixels // 2):
        y = random.randint(0, h - 1)
        x = random.randint(0, w - 1)
        noisy_image[y, x] = [0, 0, 0]

    return noisy_image

# Utility function to apply noir effect
def apply_noir(image):
# Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Enhance contrast using histogram equalization
    equalized_image = cv2.equalizeHist(gray_image)

    # Create a vignette effect
    rows, cols = equalized_image.shape
    # Create a Gaussian kernel
    X_resultant_kernel = cv2.getGaussianKernel(cols, 250)
    Y_resultant_kernel = cv2.getGaussianKernel(rows, 250)
    gaussian_kernel = Y_resultant_kernel * X_resultant_kernel.T
    # Normalize to make the kernel values between 0 and 1
    vignette_mask = gaussian_kernel / gaussian_kernel.max()
    
    # Convert vignette mask to the same type as the equalized image
    vignette_mask = cv2.normalize(vignette_mask, None, alpha=1, beta=0, norm_type=cv2.NORM_MINMAX)

    # Apply the vignette mask
    # Explicitly convert equalized_image to float32 to prevent type mismatch
    noir_image = cv2.multiply(equalized_image.astype(np.float32), vignette_mask.astype(np.float32))

    # Convert back to uint8
    noir_image = cv2.convertScaleAbs(noir_image)

    return noir_image

def has_watermark(original_image, watermarked_image, threshold=30):
    # Convert images to numpy arrays
    original = np.array(Image.open(original_image))
    watermarked = np.array(Image.open(watermarked_image))

    # Compute the absolute difference between the original and watermarked images
    difference = cv2.absdiff(original, watermarked)

    # Convert the difference image to grayscale
    gray_diff = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)

    # Apply a threshold to create a binary image of the differences
    _, thresh = cv2.threshold(gray_diff, threshold, 255, cv2.THRESH_BINARY)

    # Count non-zero pixels in the thresholded image
    non_zero_count = cv2.countNonZero(thresh)

    return non_zero_count > 0  # Returns True if watermark is detected

@app.route('/Add_Watermark', methods=['GET', 'POST'])
def add_watermark_route():
    if request.method == 'POST':
        # Get the uploaded files (image and watermark) safely
        image_file = request.files.get('image')
        watermark_file = request.files.get('watermark')

        if not image_file or not watermark_file:
            # Handle the case where one of the files is missing
            return "Please upload both the image and the watermark", 400

        # Apply the watermark
        img_with_watermark = add_watermark(image_file, watermark_file, opacity=0.4, scale=0.2)

        # Save the watermarked image temporarily for downloading
        watermarked_image_path = 'watermarked_image.png'
        with open(watermarked_image_path, 'wb') as f:
            f.write(base64.b64decode(img_with_watermark))

        # Check if the watermark is present
        watermark_exists = has_watermark(image_file, watermarked_image_path)

        return render_template('add_watermark.html',
                               original_image=image_file.filename,
                               watermarked_image=f"data:image/png;base64,{img_with_watermark}",
                               watermark_exists=watermark_exists,
                               download_link=watermarked_image_path)

    return render_template('add_watermark.html')
# Function to overlay a watermark on an image and return base64 result
def add_watermark(image, watermark, opacity=0.7, scale=0.2):
    # Convert the input images to NumPy arrays
    img = np.array(Image.open(image))
    watermark_img = np.array(Image.open(watermark))

    # Check if the images are grayscale and convert them to RGB
    if len(img.shape) == 2:  # Grayscale image (no color channels)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if len(watermark_img.shape) == 2:  # Grayscale watermark
        watermark_img = cv2.cvtColor(watermark_img, cv2.COLOR_GRAY2RGB)

    # Ensure both images are RGB (convert RGBA to RGB if necessary)
    if img.shape[2] == 4:  # If main image has alpha channel, convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    # Get the main image dimensions
    h_img, w_img = img.shape[:2]

    # Resize the watermark
    if watermark_img.shape[2] == 4:  # If watermark has an alpha channel (RGBA)
        watermark_color = watermark_img[:, :, :3]  # Get RGB
        watermark_alpha = watermark_img[:, :, 3]  # Get Alpha

        # Resize the watermark color and alpha channels
        watermark_resized = cv2.resize(watermark_color, (int(w_img * scale), int(h_img * scale)))
        alpha_resized = cv2.resize(watermark_alpha, (int(w_img * scale), int(h_img * scale)))

        # Normalize the alpha mask to [0, 1]
        alpha_resized = alpha_resized / 255.0
    else:
        # No alpha channel, use default opacity
        watermark_resized = cv2.resize(watermark_img, (int(w_img * scale), int(h_img * scale)))
        alpha_resized = np.ones_like(watermark_resized[:, :, 0]) * opacity

    # Get the watermark's dimensions after resizing
    h_wmark, w_wmark = watermark_resized.shape[:2]

    # Calculate center position for the watermark
    center_x = (w_img - w_wmark) // 2
    center_y = (h_img - h_wmark) // 2

    # Extract the region of interest (ROI) from the main image at the center
    roi = img[center_y:center_y + h_wmark, center_x:center_x + w_wmark]

    # Ensure ROI and watermark are the same size
    if roi.shape != watermark_resized.shape:
        raise ValueError("ROI and watermark sizes don't match!")

    # Blend the watermark with the ROI using the alpha channel and opacity
    for c in range(0, 3):
        roi[:, :, c] = (1 - alpha_resized) * roi[:, :, c] + alpha_resized * watermark_resized[:, :, c]

    # Place the blended watermark back into the original image
    img[center_y:center_y + h_wmark, center_x:center_x + w_wmark] = roi
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # Encode the image to PNG and convert to base64 directly
    success, encoded_image = cv2.imencode('.png', img)
    if not success:
        raise ValueError("Could not encode image")

    # Convert the encoded image to base64
    image_base64 = base64.b64encode(encoded_image).decode('utf-8')

    return image_base64


@app.route('/download')
def download():
    # Provide a download link for the watermarked image
    return send_file('image.png', as_attachment=True)


@app.route('/Inpainting', methods=['GET', 'POST'])
def inpainting_page():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            filename = save_uploaded_file(file, app.config['UPLOAD_FOLDER'])
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # Load the image
            image = cv2.imread(filepath)

            # Create mask from POST request
            if 'mask' in request.form:
                mask_data = request.form['mask']
                mask_array = np.frombuffer(base64.b64decode(mask_data), np.uint8)
                mask = cv2.imdecode(mask_array, cv2.IMREAD_GRAYSCALE)

                # Apply inpainting
                restored_image = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)

                # Convert the restored image to base64 to display on HTML
                _, buffer = cv2.imencode('.png', restored_image)
                restored_image_data = base64.b64encode(buffer).decode('ascii')

                return render_template('inpainting.html',
                                       original_image=get_image_url(filename),
                                       restored_image=f"data:image/png;base64,{restored_image_data}",
                                       mask_created=True)

    return render_template('inpainting.html')



### this is unused function for now
## fitur buat buat /bisa di akses di /Collage tidak ada di home
# Utility function to pad an image to a specific size
def pad_image(image, target_height, target_width):
    height, width = image.shape[:2]
    
    # Calculate the padding values for height and width
    pad_height = max(0, target_height - height)
    pad_width = max(0, target_width - width)

    # Pad the image with black if necessary
    padded_image = cv2.copyMakeBorder(image, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return padded_image

# Utility function to create a collage from a list of images
def create_collage(image_paths, layout='grid', order=None):
    images = [cv2.imread(path) for path in image_paths]

    # Resize all images to the smallest size
    min_height = min(image.shape[0] for image in images)
    min_width = min(image.shape[1] for image in images)

    # Resize images
    resized_images = [cv2.resize(image, (min_width, min_height)) for image in images]

    # If an order is provided, rearrange the images
    if order:
        resized_images = [resized_images[i] for i in order]

    if layout == 'grid':
        num_images = len(resized_images)
        num_cols = int(np.ceil(np.sqrt(num_images)))
        num_rows = int(np.ceil(num_images / num_cols))

        rows = []
        for i in range(0, num_images, num_cols):
            row_images = resized_images[i:i + num_cols]
            if len(row_images) < num_cols:
                blank_image = np.zeros_like(resized_images[0])
                row_images += [blank_image] * (num_cols - len(row_images))
            rows.append(np.hstack(row_images))
        collage = np.vstack(rows)

    elif layout == 'horizontal':
        max_height = max(image.shape[0] for image in resized_images)
        padded_images = [pad_image(img, max_height, img.shape[1]) for img in resized_images]
        collage = np.hstack(padded_images)

    elif layout == 'vertical':
        max_width = max(image.shape[1] for image in resized_images)
        padded_images = [pad_image(img, img.shape[0], max_width) for img in resized_images]
        collage = np.vstack(padded_images)

    elif layout == 'L-shaped':
        half = len(resized_images) // 2
        top = np.hstack(resized_images[:half])
        
        # Pad bottom images to match the width of the top row
        bottom_images = resized_images[half:]
        padded_bottom_images = [pad_image(img, min_height, top.shape[1]) for img in bottom_images]
        bottom = np.vstack(padded_bottom_images)
        collage = np.vstack([top, bottom])

    elif layout == 'checkerboard':
        num_images = len(resized_images)
        num_cols = 2  # Fixed to 2 columns for checkerboard
        num_rows = int(np.ceil(num_images / num_cols))

        rows = []
        for i in range(0, num_images, num_cols):
            row_images = resized_images[i:i + num_cols]
            if len(row_images) < num_cols:
                blank_image = np.zeros_like(resized_images[0])
                row_images += [blank_image] * (num_cols - len(row_images))
            rows.append(np.hstack(row_images))
        collage = np.vstack(rows)

    elif layout == 'diagonal-tl-br':
        rows = []
        for idx in range(len(resized_images)):
            # Create a diagonal effect with padding
            padding = [np.zeros_like(resized_images[0])] * idx
            padded_row = np.hstack(padding + [resized_images[idx]])
            rows.append(padded_row)

        # Pad the right side to ensure all rows have the same width
        max_width = max(row.shape[1] for row in rows)
        collage = np.vstack([pad_image(row, min_height, max_width) for row in rows])

    elif layout == 'diagonal-tr-bl':
        rows = []
        num_images = len(resized_images)

        # Create a blank canvas to hold the collage
        blank_row_height = min_height  # Height of each row
        blank_row_width = num_images * min_width  # Total width of the row
        collage_height = blank_row_height * num_images  # Total height for all rows

        # Initialize an empty collage with the proper height and width
        collage = np.zeros((collage_height, blank_row_width, 3), dtype=np.uint8)

        # Fill the collage with images according to the specified diagonal pattern
        for idx in range(num_images):
            # Calculate the row and column position for each image
            row_index = idx * min_height
            col_index = (num_images - 1 - idx) * min_width

            # Place the image in the correct position
            collage[row_index:row_index + min_height, col_index:col_index + min_width] = resized_images[idx]

    return collage

# Collage route
@app.route('/Collage', methods=['GET', 'POST'])
def collage_page():
    if request.method == 'POST':
        # Check if 'files' exist in the request
        if 'files' not in request.files:
            return redirect(request.url)
        
        files = request.files.getlist('files')

        # Ensure at least two files are uploaded
        if len(files) < 2:
            return render_template('collage.html', error='Please upload at least two images.')
        
        # Get the selected layout and order
        layout = request.form.get('layout')
        order_input = request.form.get('order')

        # Convert the input string to a list of integers for order
        if order_input:
            try:
                order = list(map(int, order_input.split(',')))
            except ValueError:
                return render_template('collage.html', error='Invalid order format. Please use integers separated by commas.')
        else:
            order = None  # Default to None if no order is provided

        # Save the uploaded files
        filenames = [save_uploaded_file(file, app.config['UPLOAD_FOLDER']) for file in files]
        image_paths = [os.path.join(app.config['UPLOAD_FOLDER'], filename) for filename in filenames]

        # Create the collage
        collage = create_collage(image_paths, layout, order)

        # Convert the collage to base64 for display
        collage_base64 = convert_image_to_base64(collage)

        return render_template('collage.html',
                               collage_image=f"data:image/png;base64,{collage_base64}",
                               filenames=filenames,
                               layout=layout,
                               error=None)

    return render_template('collage.html', error=None)

# Collage layout change route
@app.route('/Collage/change_layout', methods=['POST'])
def change_layout():
    layout = request.form.get('new_layout')
    filenames = request.form.get('filenames').split(',')
    order = request.form.getlist('order')  # Optional order input

    # Generate file paths again
    image_paths = [os.path.join(app.config['UPLOAD_FOLDER'], filename) for filename in filenames]

    # Rearrange order if provided
    order = list(map(int, order)) if order else None

    # Create the collage with the new layout
    collage = create_collage(image_paths, layout, order)

    # Convert collage to base64
    collage_base64 = convert_image_to_base64(collage)

    return render_template('collage.html',
                           collage_image=f"data:image/png;base64,{collage_base64}",
                           layout=layout,
                           filenames=filenames,
                           error=None)

from skimage.metrics import structural_similarity as ssim
import numpy as np

# Image Comparison route
@app.route('/Image_Comparison', methods=['GET', 'POST'])
def image_comparison_page():
    if request.method == 'POST':
        if 'file1' not in request.files or 'file2' not in request.files:
            return redirect(request.url)

        file1 = request.files['file1']
        file2 = request.files['file2']

        if file1.filename == '' or file2.filename == '':
            return redirect(request.url)

        if file1 and file2:
            filename1 = save_uploaded_file(file1, app.config['UPLOAD_FOLDER'])
            filename2 = save_uploaded_file(file2, app.config['UPLOAD_FOLDER'])
            
            filepath1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
            filepath2 = os.path.join(app.config['UPLOAD_FOLDER'], filename2)

            image1 = cv2.imread(filepath1)
            image2 = cv2.imread(filepath2)

            image1 = cv2.resize(image1, (image2.shape[1], image2.shape[0]))

            mse_value, ssim_value = compare_images(image1, image2)

            return render_template('image_comparison.html',
                                   original_image1=get_image_url(filename1),
                                   original_image2=get_image_url(filename2),
                                   mse=mse_value, ssim=ssim_value)

    return render_template('image_comparison.html')

# Function to calculate MSE and SSIM
def compare_images(image1, image2):
    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY) 
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY) 

    # Calculate MSE (Mean Squared Error)
    mse = np.mean((gray_image1 - gray_image2) ** 2)

    # Calculate SSIM (Structural Similarity Index) 
    ssim_index = ssim(gray_image1, gray_image2) 

    return mse, ssim_index

if __name__ == "__main__":
    app.run(debug=True)