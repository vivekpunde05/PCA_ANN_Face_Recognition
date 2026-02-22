import cv2
import numpy as np
import pickle
import os
from src.data_loader import load_dataset
from src.feature_extraction import apply_pca, apply_lda
from src.neural_network import train_mlp
from sklearn.model_selection import train_test_split

class LiveFaceRecognition:
    def __init__(self, model_path='models/face_recognition_model.pkl'):
        self.model_path = model_path
        self.pca = None
        self.lda = None
        self.clf = None
        self.class_names = None
        self.h = 300
        self.w = 300
        
        # Load Haar Cascade for face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
    def train_and_save_model(self, dataset_path='dataset/faces/', n_components=150):
        """Train the model and save it for live recognition"""
        print("Training model...")
        
        # Load dataset
        X, y, self.class_names, n_samples = load_dataset(dataset_path, self.h, self.w)
        
        print(f"Loaded {n_samples} samples from {len(self.class_names)} classes")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )
        
        # Apply PCA
        self.pca, eigenfaces, X_train_pca, X_test_pca = apply_pca(
            X_train, X_test, n_components, self.h, self.w
        )
        
        # Apply LDA
        self.lda, X_train_lda, X_test_lda = apply_lda(
            X_train_pca, X_test_pca, y_train
        )
        
        # Train MLP
        self.clf = train_mlp(X_train_lda, y_train)
        
        # Save model
        self._save_model()
        
        print("Model trained and saved successfully!")
        
    def _save_model(self):
        """Save the trained model components"""
        os.makedirs('models', exist_ok=True)
        
        model_data = {
            'pca': self.pca,
            'lda': self.lda,
            'clf': self.clf,
            'class_names': self.class_names,
            'h': self.h,
            'w': self.w
        }
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)
            
    def load_model(self):
        """Load the pre-trained model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. Please train the model first."
            )
        
        with open(self.model_path, 'rb') as f:
            model_data = pickle.load(f)
            
        self.pca = model_data['pca']
        self.lda = model_data['lda']
        self.clf = model_data['clf']
        self.class_names = model_data['class_names']
        self.h = model_data['h']
        self.w = model_data['w']
        
        print("Model loaded successfully!")
        
    def preprocess_face(self, face_img):
        """Preprocess detected face for recognition"""
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (self.h, self.w))
        flattened = resized.flatten().reshape(1, -1)
        return flattened
        
    def recognize_face(self, face_img):
        """Recognize a face and return name and confidence"""
        # Preprocess
        face_data = self.preprocess_face(face_img)
        
        # Transform through PCA and LDA
        face_pca = self.pca.transform(face_data)
        face_lda = self.lda.transform(face_pca)
        
        # Predict
        prediction = self.clf.predict(face_lda)[0]
        probabilities = self.clf.predict_proba(face_lda)[0]
        confidence = np.max(probabilities) * 100
        
        name = self.class_names[prediction]
        
        return name, confidence
        
    def start_live_recognition(self, confidence_threshold=50):
        """Start live webcam face recognition"""
        print("Starting live face recognition...")
        print("Press 'q' to quit")
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
            
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Could not read frame")
                break
                
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(100, 100)
            )
            
            # Process each detected face
            for (x, y, w, h) in faces:
                # Extract face region
                face_img = frame[y:y+h, x:x+w]
                
                try:
                    # Recognize face
                    name, confidence = self.recognize_face(face_img)
                    
                    # Determine color based on confidence
                    if confidence >= confidence_threshold:
                        color = (0, 255, 0)  # Green for recognized
                        label = f"{name}: {confidence:.1f}%"
                    else:
                        color = (0, 165, 255)  # Orange for low confidence
                        label = f"Unknown: {confidence:.1f}%"
                        
                except Exception as e:
                    color = (0, 0, 255)  # Red for error
                    label = "Error"
                    print(f"Recognition error: {e}")
                
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Draw label background
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(
                    frame,
                    (x, y - 30),
                    (x + label_size[0], y),
                    color,
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    frame,
                    label,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )
            
            # Display info
            info_text = f"Faces detected: {len(faces)} | Press 'q' to quit"
            cv2.putText(
                frame,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            
            # Show frame
            cv2.imshow('Live Face Recognition', frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("Live recognition stopped")


def main():
    recognizer = LiveFaceRecognition()
    
    # Check if model exists
    if not os.path.exists('models/face_recognition_model.pkl'):
        print("No trained model found. Training new model...")
        recognizer.train_and_save_model()
    else:
        print("Loading existing model...")
        recognizer.load_model()
    
    # Start live recognition
    recognizer.start_live_recognition(confidence_threshold=50)


if __name__ == "__main__":
    main()
