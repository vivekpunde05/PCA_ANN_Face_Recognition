# Usage Guide - Face Recognition System

## Getting Started

### Option 1: Web Application (Easiest) 🌐

Perfect for beginners and non-technical users.

```bash
streamlit run web_app.py
```

**What you can do:**
1. Upload photos to recognize faces
2. Train the model with one click
3. See results with confidence scores
4. View all people in your dataset

**Best for:** Testing with images, demonstrations, easy model training

---

### Option 2: Live Webcam Recognition 📹

Real-time face recognition using your computer's camera.

```bash
python live_recognition.py
```

**What happens:**
1. Model trains automatically if needed (first time only)
2. Webcam opens showing live video
3. Faces are detected and recognized in real-time
4. Green boxes = recognized, Orange = uncertain
5. Press 'q' to quit

**Best for:** Real-time applications, security systems, attendance tracking

---

### Option 3: Command Line Training 💻

For developers who want to see the technical details.

```bash
python main.py
```

**What you'll see:**
1. Dataset loading statistics
2. Eigenfaces visualization
3. Training progress
4. Accuracy metrics
5. Prediction gallery

**Best for:** Understanding the algorithm, research, customization

---

## Step-by-Step Tutorials

### Tutorial 1: First Time Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare your dataset:**
   - Create folders in `dataset/faces/` for each person
   - Add at least 20-30 images per person
   - Use clear, front-facing photos

3. **Train the model:**
   ```bash
   python main.py
   ```
   
4. **Test with webcam:**
   ```bash
   python live_recognition.py
   ```

---

### Tutorial 2: Adding New People

1. **Create a new folder:**
   ```
   dataset/faces/NewPerson/
   ```

2. **Add images:**
   - Copy 20-30 photos of the person
   - Name them: `face_1.jpg`, `face_2.jpg`, etc.

3. **Retrain the model:**
   ```bash
   python main.py
   ```
   
   Or use the web app and click "Retrain Model"

4. **Test recognition:**
   ```bash
   python live_recognition.py
   ```

---

### Tutorial 3: Using the Web App

1. **Start the app:**
   ```bash
   streamlit run web_app.py
   ```

2. **Train the model:**
   - Click "Train Model" in the sidebar
   - Wait for training to complete

3. **Upload an image:**
   - Go to "Upload Image" tab
   - Click "Browse files"
   - Select a photo with faces

4. **View results:**
   - See detected faces with bounding boxes
   - Check confidence scores
   - Expand each face for details

---

## Common Use Cases

### Use Case 1: Attendance System

```python
# Modify live_recognition.py to log attendance
# Add this after recognition:

if confidence >= 70:
    timestamp = datetime.now()
    log_attendance(name, timestamp)
```

### Use Case 2: Security System

```python
# Set high confidence threshold
recognizer.start_live_recognition(confidence_threshold=80)

# Add alert for unknown faces
if confidence < 50:
    send_alert("Unknown person detected")
```

### Use Case 3: Photo Organization

```python
# Use web_app.py to batch process photos
# Organize photos by recognized person
```

---

## Tips for Best Results

### Image Quality
- ✅ Use clear, well-lit photos
- ✅ Front-facing or slight angle
- ✅ Consistent image quality
- ❌ Avoid blurry images
- ❌ Avoid extreme angles
- ❌ Avoid heavy shadows

### Dataset Size
- Minimum: 15-20 images per person
- Recommended: 30-50 images per person
- More images = better accuracy

### Lighting
- Use consistent lighting across images
- Avoid backlighting
- Natural light works best

### Camera Setup (Live Recognition)
- Position camera at eye level
- Ensure good lighting
- Keep face 2-4 feet from camera
- Avoid cluttered backgrounds

---

## Troubleshooting

### Problem: Low Accuracy

**Solutions:**
1. Add more training images
2. Improve image quality
3. Increase PCA components to 200
4. Retrain with better lighting

### Problem: Webcam Not Opening

**Solutions:**
1. Check if webcam is connected
2. Close other apps using the camera
3. Try different camera index:
   ```python
   cap = cv2.VideoCapture(1)  # Try 1, 2, etc.
   ```

### Problem: Slow Performance

**Solutions:**
1. Reduce image size (H=150, W=150)
2. Decrease PCA components
3. Use fewer training images
4. Close other applications

### Problem: "Model Not Found"

**Solution:**
```bash
python main.py  # Train the model first
```

---

## Advanced Configuration

### Adjust Recognition Sensitivity

**High Security (Strict):**
```python
confidence_threshold = 80  # Only very confident matches
```

**Balanced:**
```python
confidence_threshold = 60  # Good balance
```

**Lenient:**
```python
confidence_threshold = 40  # More permissive
```

### Optimize for Speed

```python
# In main.py or live_recognition.py
H = 100
W = 100
N_COMPONENTS = 50
```

### Optimize for Accuracy

```python
# In main.py or live_recognition.py
H = 400
W = 400
N_COMPONENTS = 200
```

---

## Command Reference

| Command | Purpose |
|---------|---------|
| `python main.py` | Train model and show visualizations |
| `python live_recognition.py` | Start webcam recognition |
| `streamlit run web_app.py` | Launch web interface |
| `pip install -r requirements.txt` | Install dependencies |

---

## Keyboard Shortcuts

### Live Recognition
- `q` - Quit the application

### Web App
- `Ctrl + R` - Refresh the page
- `Ctrl + C` (in terminal) - Stop the server

---

## Next Steps

1. ✅ Install and test with provided dataset
2. ✅ Try live webcam recognition
3. ✅ Add your own photos
4. ✅ Retrain the model
5. ✅ Experiment with parameters
6. ✅ Build your own application

---

## Need Help?

- Check the main README.md for detailed documentation
- Review error messages carefully
- Ensure all dependencies are installed
- Verify dataset structure is correct

Happy recognizing! 🎉
