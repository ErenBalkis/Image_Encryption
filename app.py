"""
Image Encryption Tool
A Streamlit-based web application for encrypting images using various cryptographic algorithms.
Features a cyberpunk/hacker aesthetic with neon green theme.

Supported Algorithms:
- AES-CBC (AES with Cipher Block Chaining)
- AES-ECB (AES with Electronic Codebook - educational)
- DES (Data Encryption Standard)
- Vernam Cipher (One-Time Pad)
- Hill Cipher (Matrix-based encryption)

"""

import streamlit as st
import io
import base64
import hashlib
import numpy as np
from PIL import Image
from Crypto.Cipher import AES, DES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes


# ========================================
# CUSTOM CSS FOR CYBERPUNK THEME
# ========================================

def apply_custom_css():
    """Apply cyberpunk/hacker aesthetic with dark background and neon green."""
    st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background-color: #0E1117;
        color: #00FF00;
        font-family: 'Courier New', monospace;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #00FF00 !important;
        font-family: 'Courier New', monospace !important;
        text-shadow: 0 0 10px #00FF00;
    }
    
    /* Main title */
    .main-title {
        font-size: 3em;
        text-align: center;
        color: #00FF00;
        text-shadow: 0 0 20px #00FF00, 0 0 30px #00FF00;
        margin-bottom: 20px;
        font-weight: bold;
    }
    
    /* Subtitle */
    .subtitle {
        text-align: center;
        color: #00DD00;
        font-size: 1.2em;
        margin-bottom: 30px;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #001100;
        color: #00FF00;
        border: 2px solid #00FF00;
        border-radius: 5px;
        font-family: 'Courier New', monospace;
        font-weight: bold;
        padding: 10px 20px;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background-color: #00FF00;
        color: #000000;
        box-shadow: 0 0 20px #00FF00;
    }
    
    /* Download buttons */
    .stDownloadButton > button {
        background-color: #001100;
        color: #00FF00;
        border: 2px solid #00FF00;
        border-radius: 5px;
        font-family: 'Courier New', monospace;
        font-weight: bold;
        padding: 10px 20px;
        transition: all 0.3s;
    }
    
    .stDownloadButton > button:hover {
        background-color: #00FF00;
        color: #000000;
        box-shadow: 0 0 20px #00FF00;
    }
    
    /* Text inputs */
    .stTextInput > div > div > input {
        background-color: #0a0a0a;
        color: #00FF00;
        border: 1px solid #00FF00;
        font-family: 'Courier New', monospace;
    }
    
    /* Select box */
    .stSelectbox > div > div > select {
        background-color: #0a0a0a;
        color: #00FF00;
        border: 1px solid #00FF00;
        font-family: 'Courier New', monospace;
    }
    
    /* File uploader */
    .stFileUploader > div {
        background-color: #0a0a0a;
        border: 2px dashed #00FF00;
        border-radius: 5px;
        padding: 20px;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #0a0a0a;
    }
    
    /* Info/warning/error boxes */
    .stAlert {
        background-color: #0a0a0a;
        border-left: 4px solid #00FF00;
        color: #00FF00;
    }
    
    /* Divider */
    hr {
        border-color: #00FF00;
    }
    
    /* Labels */
    label {
        color: #00FF00 !important;
        font-family: 'Courier New', monospace !important;
    }
    
    /* Code blocks */
    code {
        color: #00FF00;
        background-color: #0a0a0a;
    }
    </style>
    """, unsafe_allow_html=True)


# ========================================
# UTILITY FUNCTIONS
# ========================================

def validate_and_prepare_key(key_str, algorithm):
    """
    Validate and prepare the key for the selected algorithm.
    Uses SHA-256 hashing to ensure correct key length.
    
    Args:
        key_str: User input key string
        algorithm: Selected encryption algorithm
        
    Returns:
        bytes: Prepared key of correct length
    """
    if algorithm in ["AES-CBC", "AES-ECB"]:
        # AES-256 requires 32 bytes
        if len(key_str.encode()) < 32:
            # Hash the key to get exactly 32 bytes
            return hashlib.sha256(key_str.encode()).digest()
        else:
            return key_str.encode()[:32]
    
    elif algorithm == "DES":
        # DES requires 8 bytes
        if len(key_str.encode()) < 8:
            # Hash and take first 8 bytes
            return hashlib.sha256(key_str.encode()).digest()[:8]
        else:
            return key_str.encode()[:8]
    
    elif algorithm in ["Vernam", "Hill"]:
        # Return as-is for Vernam and Hill
        return key_str.encode()
    
    return key_str.encode()


def bytes_to_image(encrypted_bytes, original_shape):
    """
    Convert encrypted bytes back to an image for visualization.
    
    Args:
        encrypted_bytes: Encrypted image data
        original_shape: Original image dimensions (height, width, channels)
        
    Returns:
        PIL.Image: Image object for display
    """
    try:
        # Calculate required size
        required_size = original_shape[0] * original_shape[1] * original_shape[2]
        
        # Pad or truncate to match original size
        if len(encrypted_bytes) < required_size:
            # Pad with zeros
            encrypted_bytes = encrypted_bytes + b'\x00' * (required_size - len(encrypted_bytes))
        elif len(encrypted_bytes) > required_size:
            # Truncate
            encrypted_bytes = encrypted_bytes[:required_size]
        
        # Convert to numpy array and reshape
        encrypted_array = np.frombuffer(encrypted_bytes, dtype=np.uint8)
        encrypted_array = encrypted_array.reshape(original_shape)
        
        # Create PIL Image
        return Image.fromarray(encrypted_array, mode='RGB')
    
    except Exception as e:
        st.error(f"Error converting bytes to image: {e}")
        # Return a blank image as fallback
        return Image.new('RGB', (original_shape[1], original_shape[0]), color='black')


def create_download_button(data, filename, label, key):
    """
    Create a download button for encrypted data.
    
    Args:
        data: Binary data to download
        filename: Suggested filename
        label: Button label
        key: Unique key for the button
    """
    st.download_button(
        label=label,
        data=data,
        file_name=filename,
        mime="application/octet-stream",
        key=key
    )


# ========================================
# ENCRYPTION ALGORITHMS
# ========================================

def encrypt_image_aes_cbc(image_bytes, key):
    """
    Encrypt image using AES in CBC mode.
    Uses a random IV for security.
    
    Args:
        image_bytes: Image data as bytes
        key: Encryption key (32 bytes for AES-256)
        
    Returns:
        tuple: (encrypted_data, iv)
    """
    try:
        # Generate random IV
        iv = get_random_bytes(16)
        
        # Create cipher
        cipher = AES.new(key, AES.MODE_CBC, iv)
        
        # Pad data to block size and encrypt
        padded_data = pad(image_bytes, AES.block_size)
        encrypted_data = cipher.encrypt(padded_data)
        
        return encrypted_data, iv
    
    except Exception as e:
        st.error(f"AES-CBC Encryption Error: {e}")
        return None, None


def encrypt_image_aes_ecb(image_bytes, key):
    """
    Encrypt image using AES in ECB mode.
    WARNING: ECB mode is insecure and included for educational purposes only.
    Patterns in the original image may still be visible.
    
    Args:
        image_bytes: Image data as bytes
        key: Encryption key (32 bytes for AES-256)
        
    Returns:
        bytes: Encrypted data
    """
    try:
        # Create cipher (no IV in ECB mode)
        cipher = AES.new(key, AES.MODE_ECB)
        
        # Pad data to block size and encrypt
        padded_data = pad(image_bytes, AES.block_size)
        encrypted_data = cipher.encrypt(padded_data)
        
        return encrypted_data
    
    except Exception as e:
        st.error(f"AES-ECB Encryption Error: {e}")
        return None


def encrypt_image_des(image_bytes, key):
    """
    Encrypt image using DES algorithm.
    Uses CBC mode with random IV.
    
    Args:
        image_bytes: Image data as bytes
        key: Encryption key (8 bytes for DES)
        
    Returns:
        tuple: (encrypted_data, iv)
    """
    try:
        # Generate random IV
        iv = get_random_bytes(8)
        
        # Create cipher
        cipher = DES.new(key, DES.MODE_CBC, iv)
        
        # Pad data to block size and encrypt
        padded_data = pad(image_bytes, DES.block_size)
        encrypted_data = cipher.encrypt(padded_data)
        
        return encrypted_data, iv
    
    except Exception as e:
        st.error(f"DES Encryption Error: {e}")
        return None, None


def encrypt_image_vernam(image_bytes, key):
    """
    Encrypt image using Vernam Cipher (One-Time Pad).
    Performs XOR operation between image bytes and key.
    
    Args:
        image_bytes: Image data as bytes
        key: Encryption key (repeated if shorter than image)
        
    Returns:
        bytes: Encrypted data
    """
    try:
        # Convert to numpy arrays for efficient XOR
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        
        # Extend key to match image length by repeating
        key_array = np.frombuffer(key, dtype=np.uint8)
        extended_key = np.tile(key_array, (len(image_array) // len(key_array)) + 1)[:len(image_array)]
        
        # XOR operation
        encrypted_array = np.bitwise_xor(image_array, extended_key)
        
        return encrypted_array.tobytes()
    
    except Exception as e:
        st.error(f"Vernam Cipher Encryption Error: {e}")
        return None


def encrypt_image_hill(image_bytes, key_matrix_size=2):
    """
    Encrypt image using Hill Cipher.
    Performs matrix multiplication modulo 256.
    
    Args:
        image_bytes: Image data as bytes
        key_matrix_size: Size of the key matrix (2 or 3)
        
    Returns:
        bytes: Encrypted data
    """
    try:
        # Generate a random invertible key matrix
        key_matrix = generate_invertible_matrix(key_matrix_size)
        
        # Convert image bytes to array
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        
        # Pad array to be divisible by matrix size
        padding_length = (key_matrix_size - (len(image_array) % key_matrix_size)) % key_matrix_size
        if padding_length > 0:
            image_array = np.append(image_array, np.zeros(padding_length, dtype=np.uint8))
        
        # Reshape into vectors of size key_matrix_size
        vectors = image_array.reshape(-1, key_matrix_size)
        
        # Encrypt each vector: C = K @ P mod 256
        encrypted_vectors = np.dot(vectors, key_matrix.T) % 256
        
        # Flatten back to 1D array
        encrypted_array = encrypted_vectors.flatten().astype(np.uint8)
        
        # Store key matrix for potential decryption (in session state)
        st.session_state.hill_key_matrix = key_matrix
        
        return encrypted_array.tobytes()
    
    except Exception as e:
        st.error(f"Hill Cipher Encryption Error: {e}")
        return None


def generate_invertible_matrix(size):
    """
    Generate a random invertible matrix modulo 256.
    
    Args:
        size: Matrix dimension (2 or 3)
        
    Returns:
        np.array: Invertible matrix
    """
    max_attempts = 100
    for _ in range(max_attempts):
        # Generate random matrix
        matrix = np.random.randint(0, 256, (size, size))
        
        # Check if determinant is coprime with 256
        det = int(np.round(np.linalg.det(matrix))) % 256
        
        if det != 0 and np.gcd(det, 256) == 1:
            return matrix
    
    # Fallback: use a known invertible matrix
    if size == 2:
        return np.array([[3, 5], [7, 11]])  # Known invertible 2x2 matrix
    else:
        return np.array([[2, 3, 5], [7, 11, 13], [17, 19, 23]])  # Known invertible 3x3 matrix


# ========================================
# MAIN APPLICATION
# ========================================

def main():
    """Main application function."""
    
    # Apply custom CSS
    apply_custom_css()
    
    # Title
    st.markdown('<div class="main-title">🔐 IMAGE ENCRYPTION TOOL 🔐</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Secure Your Images with Advanced Cryptography</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sidebar for settings
    st.sidebar.title("⚙️ ENCRYPTION SETTINGS")
    
    # Algorithm selection
    algorithm = st.sidebar.selectbox(
        "Select Algorithm",
        ["AES-CBC", "AES-ECB", "DES", "Vernam", "Hill"],
        help="Choose the encryption algorithm to use"
    )
    
    # Display algorithm info
    algorithm_info = {
        "AES-CBC": "AES with Cipher Block Chaining. Secure mode with random IV. High entropy output.",
        "AES-ECB": "AES with Electronic Codebook. ⚠️ Educational only - may show patterns!",
        "DES": "Data Encryption Standard. Legacy algorithm with 56-bit effective key.",
        "Vernam": "One-Time Pad using XOR. Theoretically unbreakable if key length = data length.",
        "Hill": "Matrix-based cipher using modular arithmetic. Educational cryptography."
    }
    
    st.sidebar.info(f"ℹ️ {algorithm_info[algorithm]}")
    
    # Key input
    if algorithm == "Hill":
        st.sidebar.markdown("### Key Matrix Size")
        matrix_size = st.sidebar.radio("Matrix Dimension", [2, 3], help="Size of the Hill Cipher key matrix")
        st.sidebar.info(f"A random {matrix_size}x{matrix_size} invertible matrix will be generated automatically.")
        key = None
    else:
        key_input = st.sidebar.text_input(
            "Encryption Key",
            type="password",
            help="Enter your secret key. Will be auto-padded if needed.",
            value="MySecretKey123"
        )
        
        if key_input:
            key = validate_and_prepare_key(key_input, algorithm)
            
            # Show key length info
            if algorithm in ["AES-CBC", "AES-ECB"]:
                st.sidebar.success(f"✓ Key prepared: 32 bytes (AES-256)")
            elif algorithm == "DES":
                st.sidebar.success(f"✓ Key prepared: 8 bytes (DES)")
            elif algorithm == "Vernam":
                st.sidebar.success(f"✓ Key length: {len(key)} bytes")
        else:
            st.sidebar.warning("⚠️ Please enter an encryption key!")
            key = None
    
    st.sidebar.markdown("---")
    
    # File upload
    st.markdown("### 📤 UPLOAD IMAGE")
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["png", "jpg", "jpeg"],
        help="Upload the image you want to encrypt"
    )
    
    if uploaded_file is not None:
        # Load and display original image
        try:
            original_image = Image.open(uploaded_file)
            
            # Convert to RGB if needed
            if original_image.mode != 'RGB':
                original_image = original_image.convert('RGB')
            
            # Store original dimensions
            original_shape = (original_image.height, original_image.width, 3)
            
            # Convert to bytes
            image_bytes = original_image.tobytes()
            
            # Display images in two columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🖼️ Original Image")
                st.image(original_image, use_container_width=True)
                st.info(f"Size: {original_image.width}x{original_image.height} pixels")
            
            # Encryption button
            encrypt_button = st.button("🔒 ENCRYPT IMAGE", use_container_width=True)
            
            if encrypt_button:
                if algorithm == "Hill" or (key is not None):
                    with st.spinner(f"Encrypting with {algorithm}..."):
                        encrypted_data = None
                        iv = None
                        
                        # Apply selected algorithm
                        if algorithm == "AES-CBC":
                            encrypted_data, iv = encrypt_image_aes_cbc(image_bytes, key)
                        
                        elif algorithm == "AES-ECB":
                            encrypted_data = encrypt_image_aes_ecb(image_bytes, key)
                        
                        elif algorithm == "DES":
                            encrypted_data, iv = encrypt_image_des(image_bytes, key)
                        
                        elif algorithm == "Vernam":
                            encrypted_data = encrypt_image_vernam(image_bytes, key)
                        
                        elif algorithm == "Hill":
                            encrypted_data = encrypt_image_hill(image_bytes, matrix_size)
                        
                        if encrypted_data is not None:
                            # Convert encrypted data to image for visualization
                            encrypted_image = bytes_to_image(encrypted_data, original_shape)
                            
                            # Display encrypted image
                            with col2:
                                st.markdown("#### 🔐 Encrypted Image")
                                st.image(encrypted_image, use_container_width=True)
                                st.success(f"✓ Encrypted with {algorithm}")
                                
                                if iv is not None:
                                    st.info(f"IV: {base64.b64encode(iv).decode()}")
                            
                            # Download section
                            st.markdown("---")
                            st.markdown("### 💾 DOWNLOAD ENCRYPTED DATA")
                            
                            download_col1, download_col2 = st.columns(2)
                            
                            with download_col1:
                                # Download as image
                                img_buffer = io.BytesIO()
                                encrypted_image.save(img_buffer, format='PNG')
                                img_buffer.seek(0)
                                
                                st.download_button(
                                    label="📥 Download Encrypted Image (PNG)",
                                    data=img_buffer,
                                    file_name=f"encrypted_{algorithm.lower()}.png",
                                    mime="image/png",
                                    key="download_image"
                                )
                            
                            with download_col2:
                                # Download as binary
                                st.download_button(
                                    label="📥 Download Raw Binary Data",
                                    data=encrypted_data,
                                    file_name=f"encrypted_{algorithm.lower()}.bin",
                                    mime="application/octet-stream",
                                    key="download_binary"
                                )
                            
                            # Display Hill Cipher key matrix if applicable
                            if algorithm == "Hill" and "hill_key_matrix" in st.session_state:
                                st.markdown("---")
                                st.markdown("### 🔑 Hill Cipher Key Matrix")
                                st.code(str(st.session_state.hill_key_matrix))
                                st.warning("⚠️ Save this matrix if you want to decrypt the image later!")
                        
                        else:
                            st.error("❌ Encryption failed. Please try again.")
                
                else:
                    st.warning("⚠️ Please enter an encryption key first!")
        
        except Exception as e:
            st.error(f"❌ Error processing image: {e}")
    
    else:
        st.info("👆 Please upload an image to get started!")
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<div style="text-align: center; color: #00DD00; font-size: 0.9em;">'
        '🛡️ Built with Python, Streamlit & PyCryptodome | Cybersecurity Education Tool 🛡️'
        '</div>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
