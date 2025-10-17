from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image

app = FastAPI()
from fastapi.responses import Response

@app.head("/")
async def head_root():
    return Response(status_code=200)

templates = Jinja2Templates(directory="templates")

# --------------------------
# Helper: Convert image to base64
# --------------------------
def to_base64(img):
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')

# --------------------------
# Helper: Add Noise
# --------------------------
def add_gaussian_noise(image):
    row, col, ch = image.shape
    mean = 0
    sigma = 20
    gauss = np.random.normal(mean, sigma, (row, col, ch)).astype('uint8')
    noisy = cv2.add(image, gauss)
    return noisy

def add_salt_pepper_noise(image, salt_prob=0.02, pepper_prob=0.02):
    noisy = np.copy(image)
    total_pixels = image.size // 3
    num_salt = int(total_pixels * salt_prob)
    num_pepper = int(total_pixels * pepper_prob)
    
    # Salt noise
    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape[:2]]
    noisy[coords[0], coords[1], :] = 255
    
    # Pepper noise
    coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape[:2]]
    noisy[coords[0], coords[1], :] = 0
    
    return noisy

def add_speckle_noise(image):
    gauss = np.random.randn(*image.shape).astype('float32')
    noisy = image + image * gauss * 0.1
    return np.clip(noisy, 0, 255).astype('uint8')

# --------------------------
# Helper: Smoothing filters
# --------------------------
def apply_smoothing(image):
    mean = cv2.blur(image, (5,5))
    gaussian = cv2.GaussianBlur(image, (5,5), 0)
    median = cv2.medianBlur(image, 5)
    bilateral = cv2.bilateralFilter(image, 9, 75, 75)
    return mean, gaussian, median, bilateral

# --------------------------
# Routes
# --------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process")
async def process_image(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    img = np.array(Image.open(BytesIO(contents)).convert("RGB"))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # ---------------------- Add Noise ----------------------
    noisy_gaussian = add_gaussian_noise(img_bgr)
    noisy_sp = add_salt_pepper_noise(img_bgr)
    noisy_speckle = add_speckle_noise(img_bgr)

    # ---------------------- Denoising / Smoothing ----------------------
    mean_smooth, gaussian_smooth, median_smooth, bilateral_smooth = apply_smoothing(img_bgr)

    # ---------------------- Sharpening ----------------------
    kernel_sharpen = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
    sharpened = cv2.filter2D(img_bgr, -1, kernel_sharpen)

    # ---------------------- Edge Detection ----------------------
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    sobel = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3))
    prewitt_x = cv2.filter2D(gray, -1, np.array([[1,0,-1],[1,0,-1],[1,0,-1]]))
    prewitt_y = cv2.filter2D(gray, -1, np.array([[1,1,1],[0,0,0],[-1,-1,-1]]))
    prewitt = cv2.addWeighted(prewitt_x, 0.5, prewitt_y, 0.5, 0)
    canny = cv2.Canny(gray, 100, 200)

    return templates.TemplateResponse("index.html", {
        "request": request,
        # Original
        "original": to_base64(img_bgr),
        # Noisy
        "noisy_gaussian": to_base64(noisy_gaussian),
        "noisy_sp": to_base64(noisy_sp),
        "noisy_speckle": to_base64(noisy_speckle),
        # Smoothing
        "mean_smooth": to_base64(mean_smooth),
        "gaussian_smooth": to_base64(gaussian_smooth),
        "median_smooth": to_base64(median_smooth),
        "bilateral_smooth": to_base64(bilateral_smooth),
        # Sharpen
        "sharpened": to_base64(sharpened),
        # Edge
        "sobel": to_base64(sobel),
        "prewitt": to_base64(prewitt),
        "canny": to_base64(canny)
    })
