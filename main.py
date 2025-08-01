import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("FR4model.h5")

fruits_class = {
    0: 'fresh_apples',
    1: 'fresh_banana',
    2: 'fresh_oranges',
    3: 'rotten_apples',
    4: 'rotten_banana',
    5: 'rotten_oranges'
}

cap = cv2.VideoCapture(0)  

def preprocess_image(frame):
    """Preprocess frame for model prediction."""
    # Resize and normalize the frame (adjust as per model's input requirements)
    resized = cv2.resize(frame, (224, 224))  # Change to your model's input size
    normalized = resized / 255.0  # Normalize pixel values to [0, 1]
    return np.expand_dims(normalized, axis=0)  # Add batch dimension

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    input_image = preprocess_image(frame)

    predictions = model.predict(input_image)
    class_id = np.argmax(predictions)  
    confidence = predictions[0][class_id]  


    label = f"{fruits_class[class_id]} ({confidence:.2f})"

    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("Fruit Freshness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
