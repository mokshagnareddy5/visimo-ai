# visimo.ai 🧠

A real-time, privacy-first facial emotion detection web application. 
**visimo.ai** uses a highly optimized TensorFlow Lite neural network to detect 7 distinct human emotions securely and directly through your browser, accessible globally via an ngrok tunnel.

## Features
- **Real-Time Detection:** Processes webcam frames asynchronously using a 64x64 `mini_XCEPTION` model (~50ms inference time).
- **Public & Scaleable:** Hosted via `Waitress` (64 threads, 1000 connection limit) and tunneled globally via `ngrok`, allowing up to 500+ users to access it concurrently via a dynamic QR code.
- **Privacy-First:** "No Data Stored" policy. Frames are processed entirely in-memory and instantly destroyed.
- **Quality Assessment:** Active warnings for low lighting, blurriness, or distance to ensure optimal prediction accuracy.

## Tech Stack
- **Frontend:** HTML5, CSS3 (Glassmorphism), Vanilla JavaScript, `MediaDevices` API
- **Backend:** Python (Flask, Waitress WSGI)
- **Machine Learning:** OpenCV (Face Detection & CLAHE), TensorFlow Lite (`fer2013_mini_XCEPTION`)

## Setup & Run

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Server**
   ```bash
   python app.py
   ```
   *On first run, it will prompt you for an ngrok authentication token to create the public tunnel.*

3. **Access Globally**
   Scan the dynamic QR code on the desktop landing page with any smartphone to instantly use the application over the public internet.

## Hackathon Innovation
This project proves that highly concurrent, accurate artificial intelligence can be run on standard consumer hardware without compromising user biometric privacy or requiring heavy front-end installations.
