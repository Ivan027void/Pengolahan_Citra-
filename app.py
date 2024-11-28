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
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

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

@app.route('/Restoration', methods=['GET', 'POST'])
def restoration_page():
    if request.method == 'POST':
        if 'file' not in request.files or 'noise_type' not in request.form or 'restoration_method' not in request.form:
            return redirect(request.url)
        
        file = request.files['file']
        noise_type = request.form['noise_type']
        restoration_method = request.form['restoration_method']
        
        if file.filename == '':
            return redirect(request.url)
            
        if file:
            # Save file using utility function
            filename = save_uploaded_file(file, app.config['UPLOAD_FOLDER'])
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # Read image
            image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            
            # Add noise based on selected type
            if noise_type == 'salt_pepper':
                # Add salt and pepper noise
                noisy_image = add_salt_pepper_noise(image, salt_prob=0.02, pepper_prob=0.02)
            elif noise_type == 'gaussian':
                # Add Gaussian noise
                noisy_image = add_gaussian_noise(image, mean=0, sigma=25)
            elif noise_type == 'speckle':
                # Add speckle noise
                noisy_image = add_speckle_noise(image, intensity=0.1)
            elif noise_type == 'periodic':
                # Add periodic noise
                noisy_image = add_periodic_noise(image, frequency=0.1)
            
            # Apply restoration method
            if noise_type == 'salt_pepper':
                if restoration_method == 'median':
                    restored_image = cv2.medianBlur(noisy_image, 3)
                elif restoration_method == 'rank_order':
                    restored_image = rank_order_filter(noisy_image, kernel_size=3)
                elif restoration_method == 'outlier':
                    restored_image = outlier_method(noisy_image, threshold=50)
                elif restoration_method == 'lowpass':
                    kernel = np.ones((3,3), np.float32) / 9
                    restored_image = cv2.filter2D(noisy_image, -1, kernel)
            
            elif noise_type == 'gaussian':
                if restoration_method == 'gaussian':
                    restored_image = cv2.GaussianBlur(noisy_image, (5,5), 0)
                elif restoration_method == 'bilateral':
                    restored_image = cv2.bilateralFilter(noisy_image, 9, 75, 75)
                elif restoration_method == 'wiener': # the best
                    restored_image = wiener_filter(noisy_image, kernel_size=5, noise_variance=0.01)
            
            elif noise_type == 'speckle':
                if restoration_method == 'median':
                    restored_image = cv2.medianBlur(noisy_image, 5)
                elif restoration_method == 'lee': # the best
                    restored_image = lee_filter(noisy_image)
                elif restoration_method == 'frost':
                    restored_image = frost_filter(noisy_image)
            
            elif noise_type == 'periodic':
                if restoration_method == 'notch':
                    restored_image = notch_filter(noisy_image, freq_cutoff=0.1)
                elif restoration_method == 'bandpass':
                    restored_image = bandpass_filter(noisy_image, low_cutoff=0.05, high_cutoff=0.15)
                elif restoration_method == 'butterworth':
                    restored_image = butterworth_filter(noisy_image, cutoff=0.1, order=2)

            # Convert images to base64 for display
            noisy_image_data = convert_image_to_base64(noisy_image)
            restored_image_data = convert_image_to_base64(restored_image)

            return render_template('restoration.html',
                                original_image=get_image_url(filename),
                                noisy_image=f"data:image/png;base64,{noisy_image_data}",
                                restored_image=f"data:image/png;base64,{restored_image_data}",
                                noise_type=noise_type,
                                restoration_method=restoration_method)
    
    return render_template('restoration.html')

# Helper functions for adding noise
def add_salt_pepper_noise(image, salt_prob, pepper_prob):
    noisy = np.copy(image)
    # Add salt noise
    salt_mask = np.random.random(image.shape) < salt_prob
    noisy[salt_mask] = 255
    # Add pepper noise
    pepper_mask = np.random.random(image.shape) < pepper_prob
    noisy[pepper_mask] = 0
    return noisy

def add_gaussian_noise(image, mean, sigma):
    gaussian = np.random.normal(mean, sigma, image.shape)
    noisy = np.clip(image + gaussian, 0, 255).astype(np.uint8)
    return noisy

def add_speckle_noise(image, intensity):
    gaussian = np.random.normal(0, intensity, image.shape)
    noisy = np.clip(image + image * gaussian, 0, 255).astype(np.uint8)
    return noisy

def add_periodic_noise(image, frequency):
    rows, cols = image.shape
    x = np.arange(cols)
    y = np.arange(rows)
    
    X, Y = np.meshgrid(x, y)
    noise = 50 * np.sin(2 * np.pi * frequency * X)
    noisy = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy

# Custom restoration filters
def median_filter(image, kernel_size=3):
    """
    Median filter implementation to reduce salt-and-pepper noise.
    
    Args:
    - image: Input grayscale image.
    - kernel_size: Size of the kernel (should be odd, e.g., 3, 5, 7).
    
    Returns:
    - Restored image after applying median filter.
    """
    return cv2.medianBlur(image, kernel_size)

def rank_order_filter(image, kernel_size):
    return cv2.medianBlur(image, kernel_size)  # Simplified version

def outlier_method(image, threshold):
    result = np.copy(image)
    mean_kernel = np.ones((3,3), np.float32) / 9
    local_mean = cv2.filter2D(image, -1, mean_kernel)
    diff = np.abs(image - local_mean)
    mask = diff > threshold
    result[mask] = local_mean[mask]
    return result

def lowpass_filter(image, kernel_size=3):
    """
    Lowpass filter using an average kernel to smooth the image.
    
    Args:
    - image: Input grayscale image.
    - kernel_size: Size of the kernel (e.g., 3x3, 5x5).
    
    Returns:
    - Restored image after applying lowpass filter.
    """
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)
    return cv2.filter2D(image, -1, kernel)

def gaussian_filter(image, kernel_size=5, sigma=1.0):
    """
    Gaussian filter implementation to reduce noise.
    
    Args:
    - image: Input grayscale image.
    - kernel_size: Size of the kernel (should be odd, e.g., 3, 5, 7).
    - sigma: Standard deviation of the Gaussian kernel.
    
    Returns:
    - Restored image after applying Gaussian filter.
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

def bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
    """
    Bilateral filter implementation to preserve edges while smoothing.
    
    Args:
    - image: Input grayscale image.
    - d: Diameter of the pixel neighborhood.
    - sigma_color: Filter sigma in color space.
    - sigma_space: Filter sigma in coordinate space.
    
    Returns:
    - Restored image after applying bilateral filter.
    """
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

import numpy as np
from scipy.signal import convolve2d

def wiener_filter(image, kernel_size=5, noise_variance=0.1, signal_variance=1.0):
    """
    Wiener filter implementation to reduce noise in the image.
    
    Args:
    - image: Input grayscale image.
    - kernel_size: Size of the kernel (should be odd, e.g., 3, 5, 7).
    - noise_variance: Variance of the noise in the image.
    - signal_variance: Variance of the signal in the image.
    
    Returns:
    - Restored image after applying Wiener filter.
    """
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size**2)
    local_mean = convolve2d(image, kernel, mode='same', boundary='symm')
    local_variance = convolve2d(image**2, kernel, mode='same', boundary='symm') - local_mean**2
    result = local_mean + (local_variance / (local_variance + noise_variance)) * (image - local_mean)
    return np.uint8(np.clip(result, 0, 255))

def lee_filter(image, window_size=7, sigma=30):
    """
    Lee filter implementation for speckle noise reduction.
    Adapts to local statistics using sliding window.
    """
    img_mean = cv2.blur(image, (window_size, window_size))
    img_sqr_mean = cv2.blur(image**2, (window_size, window_size))
    img_variance = img_sqr_mean - img_mean**2
    
    # Overall variance
    overall_variance = np.mean(img_variance)
    
    # Weight function
    weights = img_variance / (img_variance + sigma**2)
    
    # Apply filter
    restored = img_mean + weights * (image - img_mean)
    return np.uint8(restored)

def frost_filter(image, window_size=7, damping_factor=2.0):
    """
    Frost filter implementation for speckle noise reduction.
    Uses exponential kernel that varies with local statistics.
    """
    pad_size = window_size // 2
    padded = np.pad(image, pad_size, mode='reflect')
    restored = np.zeros_like(image, dtype=np.float64)

    rows, cols = image.shape
    restored = np.zeros_like(image, dtype=np.float64)
    
    # Create distance kernel
    kernel_size = window_size
    center = kernel_size // 2
    y, x = np.ogrid[-center:center+1, -center:center+1]
    distance = np.sqrt(x**2 + y**2)
    
    # Process using convolution
    for i in range(rows):
        for j in range(cols):
            # Extract local window
            window = padded[i:i+window_size, j:j+window_size]
            
            # Calculate local statistics
            local_mean = np.mean(window)
            local_variance = np.var(window)
            
            # Create adaptive kernel
            k = damping_factor * local_variance / (local_mean**2)
            kernel = np.exp(-k * distance)
            kernel = kernel / np.sum(kernel)  # Normalize
            
            # Apply filter
            restored[i, j] = np.sum(window * kernel)
    
    return np.uint8(restored)

def notch_filter(image, freq_cutoff=0.1):
    """
    Removes periodic noise using a notch filter in frequency domain.
    
    Args:
    - image: Input grayscale image.
    - freq_cutoff: Cutoff frequency for removing periodic noise.
    
    Returns:
    - Restored image.
    """
    dft = np.fft.fft2(image)
    dft_shift = np.fft.fftshift(dft)
    
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2  # Center
    
    # Create a mask with a notch at the center
    mask = np.ones((rows, cols), dtype=np.uint8)
    mask[crow-int(freq_cutoff*rows):crow+int(freq_cutoff*rows), 
         ccol-int(freq_cutoff*cols):ccol+int(freq_cutoff*cols)] = 0
    
    # Apply mask and inverse FFT
    dft_shift_filtered = dft_shift * mask
    dft_inverse = np.fft.ifftshift(dft_shift_filtered)
    restored_image = np.fft.ifft2(dft_inverse)
    
    return np.abs(restored_image).astype(np.uint8)

def bandpass_filter(image, low_cutoff=0.05, high_cutoff=0.15):
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    
    # Create a bandpass mask
    mask = np.zeros((rows, cols), dtype=np.float32)
    for u in range(rows):
        for v in range(cols):
            d = np.sqrt((u - crow)**2 + (v - ccol)**2)
            if low_cutoff * crow < d < high_cutoff * crow:
                mask[u, v] = 1
    
    # Apply mask in frequency domain
    dft = np.fft.fft2(image)
    dft_shift = np.fft.fftshift(dft)
    filtered_dft = dft_shift * mask
    dft_inverse = np.fft.ifftshift(filtered_dft)
    restored_image = np.fft.ifft2(dft_inverse)
    
    return np.abs(restored_image).astype(np.uint8)

def butterworth_filter(image, cutoff=0.1, order=2):
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    
    # Create Butterworth mask
    x = np.arange(-ccol, ccol)
    y = np.arange(-crow, crow)
    X, Y = np.meshgrid(x, y)
    distance = np.sqrt(X**2 + Y**2)
    mask = 1 / (1 + (distance / (cutoff * crow))**(2 * order))
    
    # Apply mask in frequency domain
    dft = np.fft.fft2(image)
    dft_shift = np.fft.fftshift(dft)
    filtered_dft = dft_shift * mask
    dft_inverse = np.fft.ifftshift(filtered_dft)
    restored_image = np.fft.ifft2(dft_inverse)
    
    return np.abs(restored_image).astype(np.uint8)


def extract_chain_code(image, direction_type='8_direction'):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    chain_code = []
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        for i in range(len(largest_contour) - 1):
            dx = largest_contour[i + 1][0][0] - largest_contour[i][0][0]
            dy = largest_contour[i + 1][0][1] - largest_contour[i][0][1]

            if direction_type == '4_direction':
                direction_map = {(1, 0): 0, (0, -1): 1, (-1, 0): 2, (0, 1): 3}
            elif direction_type == '8_direction':
                direction_map = {(1, 0): 0, (1, -1): 1, (0, -1): 2, (-1, -1): 3,
                                 (-1, 0): 4, (-1, 1): 5, (0, 1): 6, (1, 1): 7}
            chain_code.append(direction_map.get((dx, dy), None))
    return [c for c in chain_code if c is not None]

def apply_transformation(image, transformation):
    if transformation == 'flip_horizontal':
        return cv2.flip(image, 1)
    elif transformation == 'flip_vertical':
        return cv2.flip(image, 0)
    elif transformation == 'rotate_90':
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif transformation == 'rotate_180':
        return cv2.rotate(image, cv2.ROTATE_180)
    return image

def normalize_chain_code(chain_code, num_directions):
    differences = [(chain_code[i] - chain_code[i - 1]) % num_directions for i in range(1, len(chain_code))]
    differences.append((chain_code[0] - chain_code[-1]) % num_directions)
    normalized_code = min(differences[i:] + differences[:i] for i in range(len(differences)))
    return normalized_code, differences

@app.route('/Chaincode', methods=['GET', 'POST'])
def chain_code_page():
    if request.method == 'POST':
        if 'file' not in request.files or 'transformation' not in request.form or 'chain_code_type' not in request.form:
            return redirect(request.url)

        file = request.files['file']
        transformation = request.form['transformation']
        chain_code_type = request.form['chain_code_type']

        if file.filename == '':
            return redirect(request.url)

        if file:
            filename = save_uploaded_file(file, app.config['UPLOAD_FOLDER'])
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image = cv2.imread(filepath)

            transformed_image = apply_transformation(image, transformation)
            chain_code_original = extract_chain_code(image, chain_code_type)
            chain_code_transformed = extract_chain_code(transformed_image, chain_code_type)

            are_chain_codes_same = chain_code_original == chain_code_transformed
            
            if chain_code_type == '4_direction':
                num_directions = 4
            elif chain_code_type == '8_direction':
                num_directions = 8
                
            normalized_code_original, differenced_original = normalize_chain_code(chain_code_original, num_directions)
            normalized_code_transformed, differenced_transformed = normalize_chain_code(chain_code_transformed, num_directions)

            are_normalized_same = normalized_code_original == normalized_code_transformed

            original_image_data = convert_image_to_base64(image)
            transformed_image_data = convert_image_to_base64(transformed_image)

            return render_template('chaincode.html',
                                   original_image=f"data:image/png;base64,{original_image_data}",
                                   transformed_image=f"data:image/png;base64,{transformed_image_data}",
                                   chain_code_original=chain_code_original,
                                   chain_code_transformed=chain_code_transformed,
                                   are_chain_codes_same=are_chain_codes_same,
                                   normalized_code_original=normalized_code_original,
                                   normalized_code_transformed=normalized_code_transformed,
                                   differences_original=differenced_original,
                                   differences_transformed=differenced_transformed,
                                   are_normalized_same=are_normalized_same
            )
    return render_template('chaincode.html')





# Function to resize image with interpolation
def resize_image_with_interpolation(image, scale_factor, interpolation_method):
    new_width = int(image.shape[1] * scale_factor)
    new_height = int(image.shape[0] * scale_factor)
    dim = (new_width, new_height)
    interpolation_map = {
        'bilinear': cv2.INTER_LINEAR,
        'bicubic': cv2.INTER_CUBIC,
        'nearest': cv2.INTER_NEAREST,
        'lanczos': cv2.INTER_LANCZOS4
    }
    interpolation = interpolation_map.get(interpolation_method, cv2.INTER_LINEAR)
    resized_image = cv2.resize(image, dim, interpolation=interpolation)
    return resized_image

# Function to calculate pixel difference between two images
def calculate_pixel_difference(image1, image2):
    # Convert images to float32 for more precise calculations
    image1_float = np.float32(image1)
    image2_float = np.float32(image2)
    
    # Calculate the absolute difference
    diff = np.abs(image1_float - image2_float)
    
    # Normalize the difference for visualization
    diff_normalized = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    
    # Convert the difference to an 8-bit image
    diff_image = np.uint8(diff_normalized)
    
    # Apply color map to the difference image for better visualization
    heatmap = cv2.applyColorMap(diff_image, cv2.COLORMAP_JET)
    
    return heatmap

# Function to generate a comparison plot of the selected interpolation method and pixel differences
def generate_interpolation_comparison_plot(image_path, scale_factor, selected_method):
    original_image = cv2.imread(image_path)
    interpolation_methods = ['bilinear', 'bicubic', 'nearest', 'lanczos']
    
    # Make sure the selected method is in the interpolation list
    if selected_method not in interpolation_methods:
        return None
    
    # Initialize lists to hold the resized images and difference images
    resized_images = []
    diff_images = []  # To store the difference heatmaps
    
    # Resize the image using the selected interpolation method and others
    for method in interpolation_methods:
        resized_image = resize_image_with_interpolation(original_image, scale_factor, method)
        resized_images.append(resized_image)
    
    # Calculate pixel differences between the selected method and the others
    selected_index = interpolation_methods.index(selected_method)
    
    for i, method in enumerate(interpolation_methods):
        if i != selected_index:  # Skip the selected method itself
            diff_image = calculate_pixel_difference(resized_images[selected_index], resized_images[i])
            diff_images.append(diff_image)
    
    # Plot the resized images and their pixel difference heatmaps
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    # Plot the pixel difference heatmaps
    diff_titles = [f"Diff: {selected_method.capitalize()} vs. {interpolation_methods[i]}" for i in range(len(resized_images)) if i != selected_index]
    for i, ax in enumerate(axes[:3].flatten()):
        if i < len(diff_images):
            ax.imshow(cv2.cvtColor(diff_images[i], cv2.COLOR_BGR2RGB))
            ax.set_title(diff_titles[i])
            ax.axis('off')
            
    
    # Save the plot to a BytesIO buffer
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    return plot_base64

@app.route('/Image_Interpolation', methods=['GET', 'POST'])
def image_interpolation():
    if request.method == 'POST':
        file = request.files['file']
        if not file:
            return "No file uploaded.", 400

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        scale_factor = float(request.form.get('scale_factor'))
        selected_method = request.form.get('interpolation', 'bilinear')
        compare_all = 'compare_all' in request.form  # Checkbox to compare the selected method with others

        # Generate comparison plot if "Compare All" is selected
        pixel_comparison_plot = None
        if compare_all:
            pixel_comparison_plot = generate_interpolation_comparison_plot(file_path, scale_factor, selected_method)

        # Resize with the selected interpolation method if "Compare All" is not selected
        resized_image_base64 = None
        if not compare_all:
            original_image = cv2.imread(file_path)
            resized_image = resize_image_with_interpolation(original_image, scale_factor, selected_method)
            resized_image_base64 = convert_image_to_base64(resized_image)

        return render_template(
            'interpolation.html',
            original_image=url_for('static', filename='uploads/' + filename),
            resized_image=f"data:image/png;base64,{resized_image_base64}" if resized_image_base64 else None,
            pixel_comparison_plot=f"data:image/png;base64,{pixel_comparison_plot}" if pixel_comparison_plot else None,
        )

    return render_template('interpolation.html')


# Image morphological operations route
@app.route('/Morphology', methods=['GET', 'POST'])
def morpholgy_page():
    if request.method == 'POST':
        if 'file' not in request.files or 'operation' not in request.form:
            return redirect(request.url)
        file = request.files['file']
        operation = request.form['operation']  # Erosion, Dilation, Opening, Closing
        if file.filename == '':
            return redirect(request.url)
        if file:
            # Save file using utility function
            filename = save_uploaded_file(file, app.config['UPLOAD_FOLDER'])
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # Read image
            image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

            # Apply the selected morphological operation
            if operation == 'erosion':
                kernel = np.ones((5, 5), np.uint8)
                morph_image = cv2.erode(image, kernel, iterations=1)
            elif operation == 'dilation':
                kernel = np.ones((5, 5), np.uint8)
                morph_image = cv2.dilate(image, kernel, iterations=1)
            elif operation == 'opening':
                kernel = np.ones((5, 5), np.uint8)
                morph_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
            elif operation == 'closing':
                kernel = np.ones((5, 5), np.uint8)
                morph_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

            # Convert morphological image to base64 to display on HTML
            morph_image_data = convert_image_to_base64(morph_image)

            return render_template('morphology.html',
                                   original_image=get_image_url(filename),
                                   morph_image=f"data:image/png;base64,{morph_image_data}",
                                   operation=operation)
    return render_template('morphology.html')

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
    highlight_mask = split_tone_image.mean(axis=2) > 0.5  # Mean to find highlights, hightlight color (0, 195, 255) is red
    shadow_mask = split_tone_image.mean(axis=2) <= 0.5  # Mean to find shadows, shadow color (0, 100, 0) is green

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

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the image
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    
    return jsonify({'file_path': file_path})

# Route to render the HTML page for Inpainting
@app.route('/Inpaint_Restoration', methods=['GET'])
def inpaint():
    return render_template('inpaint.html')

@app.route('/inpaint', methods=['POST'])
def inpaint_image():
    data = request.get_json()
    image_path = data.get('image_path')
    mask_data = data.get('mask')

    if not image_path or not mask_data:
        return jsonify({'error': 'Missing image or mask data'}), 400

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        return jsonify({'error': 'Failed to read image'}), 400

    # Convert mask_data (which is a 2D list) to a NumPy array
    mask = np.array(mask_data, dtype=np.uint8)

    # Resize the mask to match the image
    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))

    # Apply inpainting
    restored_image = cv2.inpaint(image, mask_resized, 3, cv2.INPAINT_TELEA)

    # Save the restored image
    restored_filename = 'restored_' + os.path.basename(image_path)
    result_path = os.path.join(UPLOAD_FOLDER, restored_filename)
    cv2.imwrite(result_path, restored_image)

    return jsonify({'restored_image': url_for('static', filename=f'uploads/{restored_filename}')})

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
    return send_file('watermarked_image.png', as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)