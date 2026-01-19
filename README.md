# 🔒 Visual Image Encryption Tool

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Framework-FF4B4B)
![Security](https://img.shields.io/badge/Security-Cryptography-green)

> **A comprehensive, educational web application for visualizing cryptographic algorithms on images, wrapped in a striking Cyberpunk/Hacker aesthetic.**

## 🎯 Project Overview

This project is a **single-file Streamlit application** designed to demonstrate the effects of various encryption algorithms on visual data. It allows users to upload images, apply cryptographic techniques (like AES, DES, Hill Cipher), and instantly visualize the results.

The tool is built with a strong focus on **Cyberpunk UI design** (Neon Green on Dark Background) and serves as an educational platform to understand concepts like "Confusion", "Diffusion", and why standard modes like ECB are insecure for image data.

---

## 🚀 Live Demo

You can try the live version of the app here:
[**🔗 Image_Encryption_Tool**](https://image-encryption-tool4you.streamlit.app/) 

---

## ✨ Key Features

### 🔐 Cryptographic Algorithms
* **✅ AES-CBC (Recommended):** Industry-standard encryption using a random IV. Produces high-entropy noise (complete visual scrambling).
* **⚠️ AES-ECB (Educational):** Demonstrates security weaknesses by preserving patterns in the encrypted image (e.g., the "Penguin" effect).
* **🗝️ DES:** Legacy encryption implementation for historical comparison.
* **✖️ Vernam Cipher (One-Time Pad):** Perfect secrecy implementation using XOR operations.
* **mjvx Hill Cipher:** Classical matrix-based encryption with automatic invertibility checking (2x2 and 3x3 matrices).

### 🎨 Cyberpunk UI / UX
* **Themed Interface:** Custom CSS implementing a `#0E1117` dark mode with `#00FF00` neon green accents.
* **Monospace Typography:** Terminal-style fonts for an immersive "hacker" experience.
* **Real-time Visualization:** Side-by-side comparison of Original vs. Encrypted images.
* **Interactive Controls:** Intuitive sidebar for algorithm selection, key input, and matrix configuration.

### 🛠️ Technical Capabilities
* **Image Processing:** Automatic RGB conversion and byte-level manipulation using `Pillow` and `NumPy`.
* **Download Options:** Users can download the encrypted result as a visual image (PNG) or raw binary data.
* **Robust Error Handling:** Automatic key padding (SHA-256), matrix invertibility checks, and format validation.

---


## 📦 Installation & Local Usage

To run this project locally, follow these steps:

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/image-encryption-tool.git](https://github.com/your-username/image-encryption-tool.git)
cd image-encryption-tool

### 2. Install Dependencies

Make sure you have Python installed. Then run:

```bash
pip install -r requirements.txt

```

*Dependencies: `streamlit`, `pycryptodome`, `numpy`, `pillow*`

### 3. Run the App

```bash
streamlit run app.py

```

The application will open in your browser at `http://localhost:8501`.

---

## 📚 Educational Value

This tool is specifically designed for **Information Security** and **Cryptography** courses to demonstrate:

| Concept | Demonstration in App |
| --- | --- |
| **ECB vs. CBC** | Users can visually compare how AES-ECB reveals shapes (low diffusion) vs. AES-CBC (high diffusion). |
| **Initialization Vector (IV)** | Shows how IVs are generated and used to randomize output in CBC mode. |
| **Visual Cryptography** | Visual proof of "High Entropy" noise generation in modern algorithms. |
| **Matrix Operations** | Practical application of Linear Algebra (Matrix Multiplication/Inversion) in the Hill Cipher module. |

---

## 📂 Project Structure

```
├── app.py                # The main application file (Logic + UI)
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation

```

---

<p align="center">
  <img src="screenshots/image-enc-1.png" width="45%" />
  <img src="screenshots/encrypted.png" width="45%" />
</p>
file:///home/eren/Resimler/image-enc-1.png

---

## 🔮 Future Improvements

* [ ] Decryption module for authorized users.
* [ ] PSNR & SSIM metrics to measure image distortion mathematically.
* [ ] Steganography features (hiding data within images).
* [ ] Support for more formats (BMP, TIFF).
