const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
let drawing = false;
let imagePath = '';
let mask = [];

// Handle form submission to upload the image
document.getElementById('uploadForm').onsubmit = async function (e) {
    e.preventDefault();
    const fileInput = document.getElementById('imageInput').files[0];
    const formData = new FormData();
    formData.append('file', fileInput);

    const response = await fetch('/upload', {
        method: 'POST',
        body: formData
    });

    const data = await response.json();
    imagePath = data.file_path;

    const img = new Image();
    img.src = imagePath;
    img.onload = function () {
        document.getElementById('originalImage').src = imagePath;
        document.getElementById('originalImage').style.display = 'block';
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);
    };

    document.getElementById('originalImage').src = imagePath;
    document.getElementById('originalImage').style.display = 'block';
    document.getElementById('processButton').style.display = 'block';
};

// Drawing the mask on the canvas
canvas.onmousedown = function (e) {
    drawing = true;
    ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
    ctx.beginPath();
};

canvas.onmousemove = function (e) {
    if (drawing) {
        const x = e.offsetX;
        const y = e.offsetY;
        ctx.arc(x, y, 10, 0, 2 * Math.PI);
        ctx.fill();

        // Store the mask data as circles with coordinates
        mask.push([x, y, 10]);
    }
};

canvas.onmouseup = function () {
    drawing = false;
};

// Process inpainting
document.getElementById('processButton').onclick = async function () {
    const response = await fetch('/inpaint', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            image_path: imagePath,
            mask: createMaskData()
        })
    });

    const data = await response.json();
    document.getElementById('restoredImage').src = data.restored_image;
    document.getElementById('output-section').style.display = 'block';
};

// Convert the drawn mask into a 2D array for processing
function createMaskData() {
    const maskArray = Array.from(Array(canvas.height), () => Array(canvas.width).fill(0));
    mask.forEach(point => {
        const [x, y, radius] = point;
        for (let i = -radius; i <= radius; i++) {
            for (let j = -radius; j <= radius; j++) {
                if (x + i >= 0 && x + i < canvas.width && y + j >= 0 && y + j < canvas.height) {
                    maskArray[y + j][x + i] = 255;  // White (255) indicates the mask area
                }
            }
        }
    });
    return maskArray;
}
