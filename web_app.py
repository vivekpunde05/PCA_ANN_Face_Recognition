import streamlit as st
import cv2
import numpy as np
import pickle
import os
from PIL import Image
from src.data_loader import load_dataset
from src.feature_extraction import apply_pca, apply_lda
from src.neural_network import train_mlp
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Face Recognition System",
    page_icon="👤",
    layout="wide"
)

class FaceRecognitionApp:
    def __init__(self):
        self.model_path = 'models/face_recognition_model.pkl'
        self.h = 300
        self.w = 300
        
        # Load face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
    def load_model(self):
        """Load the trained model"""
        if not os.path.exists(self.model_path):
            return None
            
        with open(self.model_path, 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    
    def train_model(self, dataset_path='dataset/faces/', n_components=150):
        """Train and save the model"""
        with st.spinner('Training model... This may take a few minutes.'):
            # Load dataset
            X, y, class_names, n_samples = load_dataset(dataset_path, self.h, self.w)
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42
            )
            
            # Apply PCA
            pca, eigenfaces, X_train_pca, X_test_pca = apply_pca(
                X_train, X_test, n_components, self.h, self.w
            )
            
            # Apply LDA
            lda, X_train_lda, X_test_lda = apply_lda(
                X_train_pca, X_test_pca, y_train
            )
            
            # Train MLP
            clf = train_mlp(X_train_lda, y_train)
            
            # Save model
            os.makedirs('models', exist_ok=True)
            model_data = {
                'pca': pca,
                'lda': lda,
                'clf': clf,
                'class_names': class_names,
                'h': self.h,
                'w': self.w
            }
            
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
                
            return model_data
    
    def recognize_face(self, image, model_data):
        """Recognize faces in an image"""
        # Convert PIL to OpenCV format
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50)
        )
        
        results = []
        
        for (x, y, w, h) in faces:
            # Extract and preprocess face
            face_img = img_bgr[y:y+h, x:x+w]
            gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray_face, (self.h, self.w))
            flattened = resized.flatten().reshape(1, -1)
            
            # Transform and predict
            face_pca = model_data['pca'].transform(flattened)
            face_lda = model_data['lda'].transform(face_pca)
            
            prediction = model_data['clf'].predict(face_lda)[0]
            probabilities = model_data['clf'].predict_proba(face_lda)[0]
            confidence = np.max(probabilities) * 100
            
            name = model_data['class_names'][prediction]
            
            results.append({
                'bbox': (x, y, w, h),
                'name': name,
                'confidence': confidence
            })
            
            # Draw on image
            color = (0, 255, 0) if confidence >= 50 else (255, 165, 0)
            cv2.rectangle(img_bgr, (x, y), (x+w, y+h), color, 3)
            
            label = f"{name}: {confidence:.1f}%"
            cv2.putText(
                img_bgr,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )
        
        # Convert back to RGB for display
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        return img_rgb, results


def main():
    st.title("👤 Face Recognition System")
    st.markdown("---")
    
    app = FaceRecognitionApp()
    
    # Sidebar
    st.sidebar.title("Settings")
    
    # Check if model exists
    model_data = app.load_model()
    
    if model_data is None:
        st.sidebar.warning("⚠️ No trained model found")
        if st.sidebar.button("🎓 Train Model"):
            model_data = app.train_model()
            if model_data:
                st.sidebar.success("✅ Model trained successfully!")
                st.rerun()
    else:
        st.sidebar.success("✅ Model loaded")
        st.sidebar.info(f"**Classes:** {len(model_data['class_names'])}")
        
        with st.sidebar.expander("View Classes"):
            for i, name in enumerate(model_data['class_names'], 1):
                st.write(f"{i}. {name}")
        
        if st.sidebar.button("🔄 Retrain Model"):
            model_data = app.train_model()
            if model_data:
                st.sidebar.success("✅ Model retrained!")
                st.rerun()
    
    # Main content
    if model_data is None:
        st.info("👈 Please train the model first using the sidebar")
        return
    
    tab1, tab2, tab3 = st.tabs(["📷 Upload Image", "📹 Webcam", "ℹ️ About"])
    
    with tab1:
        st.header("Upload Image for Recognition")
        
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png']
        )
        
        if uploaded_file is not None:
            # Load image
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, use_container_width=True)
            
            with col2:
                st.subheader("Recognition Results")
                
                with st.spinner('Recognizing faces...'):
                    result_img, results = app.recognize_face(image, model_data)
                
                st.image(result_img, use_container_width=True)
                
                if results:
                    st.success(f"✅ Detected {len(results)} face(s)")
                    
                    for i, result in enumerate(results, 1):
                        with st.expander(f"Face {i}"):
                            st.write(f"**Name:** {result['name']}")
                            st.write(f"**Confidence:** {result['confidence']:.2f}%")
                            
                            if result['confidence'] >= 70:
                                st.success("High confidence")
                            elif result['confidence'] >= 50:
                                st.warning("Medium confidence")
                            else:
                                st.error("Low confidence")
                else:
                    st.warning("⚠️ No faces detected in the image")
    
    with tab2:
        st.header("Webcam Recognition")
        st.info("💡 Use the live_recognition.py script for real-time webcam recognition")
        st.code("python live_recognition.py", language="bash")
        
        st.markdown("""
        The webcam feature provides:
        - Real-time face detection
        - Live recognition with confidence scores
        - Color-coded bounding boxes
        - Press 'q' to quit
        """)
    
    with tab3:
        st.header("About This System")
        
        st.markdown("""
        ### Face Recognition Pipeline
        
        This system uses a multi-stage approach:
        
        1. **Face Detection**: Haar Cascade classifier detects faces in images
        2. **Preprocessing**: Faces are resized and converted to grayscale
        3. **PCA**: Reduces dimensionality and extracts eigenfaces
        4. **LDA**: Optimizes features for class discrimination
        5. **MLP Classifier**: Neural network performs final classification
        
        ### Features
        
        - ✅ Upload images for recognition
        - ✅ Real-time webcam recognition
        - ✅ Confidence scores for predictions
        - ✅ Multiple face detection
        - ✅ Model training and retraining
        
        ### Performance
        
        Recognition accuracy depends on:
        - Dataset quality and size
        - Lighting conditions
        - Face orientation
        - Image resolution
        """)
        
        st.markdown("---")
        st.markdown("Built with Streamlit, OpenCV, and scikit-learn")


if __name__ == "__main__":
    main()
