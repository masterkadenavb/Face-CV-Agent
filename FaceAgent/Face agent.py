# -*- coding: utf-8 -*-
"""
# Image > Convert to GrayScale > detect faces > Boundary Box > Features

Quality check of the face detection (count, brightness, location, number of faces)

"""

import cv2 as cv
import cv2
import numpy as np
import sys
import json
import uuid
import time


start_time = time.time()
image_path = sys.argv[1] if len(sys.argv) > 1 else "FaceTest.png"
image = cv2.imread(image_path)
request_id = f"demo-face-{uuid.uuid4().hex[:6]}"

# ==========================================
# If no file is found
# ==========================================
if image is None:
    print(json.dumps({
        "schema_version": "1.0.0",
        "request_id": request_id,
        "status": "invalid_input",
        "face_count": 0,
        "faces": [],
        "capture_readiness": "not_ready",
        "processing_ms": int((time.time() - start_time) * 1000),
        "warnings": ["file_not_found_or_invalid_format"]
    }, indent=4))
    sys.exit(1)

# ==========================================
# Image quality checks (Blur and Darkness)
# ==========================================
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
avg_brightness = np.mean(gray)

is_blurry = laplacian_var < 40.0
is_dark = avg_brightness < 30.0

if is_blurry or is_dark:
    warnings = []
    if is_blurry: warnings.append("image_blurry")
    if is_dark: warnings.append("insufficient_illumination")

    print(json.dumps({
        "schema_version": "1.0.0",
        "request_id": request_id,
        "status": "recapture_needed",
        "face_count": 0,
        "faces": [],
        "capture_readiness": "not_ready",
        "processing_ms": int((time.time() - start_time) * 1000),
        "warnings": warnings
    }, indent=4))
    sys.exit(0)





#load the facedetector cascade classifier
cascade = cv.data.haarcascades + "haarcascade_frontalface_default.xml"
face_detector = cv.CascadeClassifier(cascade)



# Convert image to GrayScale
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)


# Detect Faces
faces = face_detector.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(80, 80)
    )

#bounding Box
img_height = image.shape[0]
img_width = image.shape[1]


processedfaces = []

#bluramount
blur_score = round(min(1.0, laplacian_var / 100.0), 2)


for (x,y,w,h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)

    norm_x = round(float(x) / img_width, 2)

    norm_x = round(float(x) / img_width, 2)
    norm_y = round(float(y) / img_height, 2)
    norm_w = round(float(w) / img_width, 2)
    norm_h = round(float(h) / img_height, 2)

    # centre of images:
    image_center_x = img_width / 2.0
    image_center_y = img_height / 2.0
    face_center_x = x + (w / 2.0)
    face_center_y = y + (h / 2.0)

    # Bounding Box Meta data
    face_area = w * h
    image_area = img_width * img_height
    size_score = round(face_area / image_area, 2)

    # max distance
    distance = abs(face_center_x - image_center_x) + abs(face_center_y - image_center_y)
    max_distance = image_center_x + image_center_y
    centering_amount= round(max(0.0, 1.0 - (distance / max_distance)), 2)

    face_data = {
        "bounding_box_normalized": {
            "x": norm_x,
            "y": norm_y,
            "width": norm_w,
            "height": norm_h
        },
        "centering_score": centering_amount,
        "size_score": size_score,
        "blur_amount": blur_score

    }
    processedfaces.append(face_data)




#face count Quality check
face_count = len(faces)
#print(f"Face Count: {face_count}")

processing_time = int((time.time() - start_time) * 1000)

if face_count == 0:
    print(json.dumps({
        "schema_version": "1.0.0",
        "request_id": request_id,
        "status": "recapture_needed",
        "face_count": 0,
        "faces": [],
        "capture_readiness": "not_ready",
        "processing_ms": processing_time,
        "warnings": ["no_face_detected"]
    }, indent=4))

elif face_count > 1:
    print(json.dumps({
        "schema_version": "1.0.0",
        "request_id": request_id,
        "status": "recapture_needed",
        "face_count": face_count,
        "faces": [],
        "capture_readiness": "not_ready",
        "processing_ms": processing_time,
        "warnings": [
            "multiple_face_like_regions_detected",
            "please_use_one_participant_at_a_time"
        ]
    }, indent=4))

else:
    # Exactly 1 face (Success)
    print(json.dumps({
        "schema_version": "1.0.0",
        "request_id": request_id,
        "status": "ok",
        "face_count": 1,
        "faces": processedfaces,
        "capture_readiness": "ready",
        "processing_ms": processing_time,
        "warnings": []
    }, indent=4))




cv2.imshow("Face Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
