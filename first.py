import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array

model = tf.keras.models.load_model('fake.h5') 

img_height, img_width = 224, 224 

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    resized_frame = cv2.resize(frame, (img_width, img_height))

    frame_array = img_to_array(resized_frame)
    frame_array = np.expand_dims(frame_array, axis=0) 
    frame_array = frame_array / 255.0 

    prediction = model.predict(frame_array)[0][0]  


    confidence = prediction * 100  
    label = "Real" if confidence > 50 else "AI"

    # Display result on frame
    text = f"{label}: {confidence:.2f}%"
    cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Real-Time Classification", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()