# 🖼️ Image Enhancer App (FastAPI + OpenCV)

A simple web application built with **FastAPI**, **OpenCV**, and **Jinja2** to enhance image quality directly from your browser.

You can upload any color image, and the app will:
1. Apply **Denoising / Smoothing** filters (mean, median, gaussian).
2. Perform **Sharpening** for clearer details.
3. Detect edges using **Sobel**, **Prewitt**, and **Canny** filters.
4. Display all processed results side-by-side.

---

## 🚀 Features

- 🧠 Multiple smoothing filters (Mean, Median, Gaussian)
- ✨ Sharpening filter for enhanced clarity
- 🔍 Edge detection with Sobel, Prewitt, and Canny operators
- 🌐 FastAPI backend + Jinja2 frontend
- ⚙️ Ready for deploy on Render

---

## 🛠️ Installation & Run Locally

1. Clone this repository
2. Create a virtual environment:
3. Install dependencies: pip install -r requirements.txt
4. Run the app: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
5. Open your browser and go to: http://127.0.0.1:8000


