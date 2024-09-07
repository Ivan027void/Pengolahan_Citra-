import os
from flask import url_for, current_app
from werkzeug.utils import secure_filename

def save_uploaded_file(file, upload_folder):
    """
    Save the uploaded file to the specified upload folder and return the filename.
    """
    filename = secure_filename(file.filename)
    filepath = os.path.join(upload_folder, filename)
    file.save(filepath)
    return filename

def get_image_url(filename, subdirectory='uploads'):
    """
    Generate a URL for the uploaded image to be used in templates.
    """
    return url_for('static', filename=f'{subdirectory}/{filename}')
