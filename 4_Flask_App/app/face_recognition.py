import numpy as np
import sqlite3
import pickle
import cv2
import os

# Load all models
haar = cv2.CascadeClassifier('./model/haarcascade_frontalface_default.xml')  # Face detector
model_svm = pickle.load(open('./model/model_svm.pickle', 'rb'))  # SVM Model
pca_models = pickle.load(open('./model/pca_dict.pickle', 'rb'))  # PCA Dictionary
model_pca = pca_models['pca']  # PCA model
mean_face_arr = pca_models['mean_face']  # Mean Face

# Database connection
def init_db():
    conn = sqlite3.connect("faces.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS unknown_faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT,
            detection_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def save_unknown_face(image, count):
    folder = "./static/unknown_faces/"
    if not os.path.exists(folder):
        os.makedirs(folder)
    filename = f"unknown_{count}.jpg"
    path = os.path.join(folder, filename)
    cv2.imwrite(path, image)
    conn = sqlite3.connect("faces.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO unknown_faces (image_path) VALUES (?)", (path,))
    conn.commit()
    conn.close()
    return filename

def faceRecognitionPipeline(filename, path=True):
    if path:
        img = cv2.imread(filename)  # Read Image
    else:
        img = filename  # If already an array
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    faces = haar.detectMultiScale(gray, 1.5, 3)  # Detect faces
    predictions = []
    unknown_count = 0
    
    for x, y, w, h in faces:
        roi = gray[y:y+h, x:x+w]
        roi = roi / 255.0  # Normalize
        roi_resize = cv2.resize(roi, (100, 100), cv2.INTER_AREA)
        roi_reshape = roi_resize.reshape(1, 10000)
        roi_mean = roi_reshape - mean_face_arr  # Subtract mean face
        eigen_image = model_pca.transform(roi_mean)
        eig_img = model_pca.inverse_transform(eigen_image)
        results = model_svm.predict(eigen_image)
        prob_score = model_svm.predict_proba(eigen_image)
        prob_score_max = prob_score.max()
        
        if prob_score_max < 0.7:  # If face not recognized
            results[0] = "Unknown"
            unknown_filename = save_unknown_face(img[y:y+h, x:x+w], unknown_count)
            unknown_count += 1
        
        text = f"{results[0]} : {int(prob_score_max * 100)}%"
        color = (0, 255, 0) if results[0] != "Unknown" else (0, 0, 255)  # Green for known, Red for unknown
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
        predictions.append({
            'roi': roi,
            'eig_img': eig_img,
            'prediction_name': results[0],
            'score': prob_score_max
        })
    
    return img, predictions
