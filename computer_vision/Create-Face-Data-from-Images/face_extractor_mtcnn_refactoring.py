from matplotlib import pyplot as plt
from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
import os
import time

start_time = time.time()

detector = MTCNN()

# extract a single face from a given photograph
def extract_face(filename, required_size=(224, 224)):
    # load image from file
    pixels = plt.imread(filename)
    # create the detector, using default weights
    # detect faces in the image
    results = detector.detect_faces(pixels)
    
    if results:  # If a face is detected
        # extract the bounding box from the first face
        x1, y1, width, height = results[0]['box']
        x2, y2 = x1 + width, y1 + height
        # extract the face
        face = pixels[y1:y2, x1:x2]
        # resize pixels to the model size
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = asarray(image)
        return face_array
    else:
        return None

# Folder containing the images
image_folder = "images"
# List to store the face images
#face_images = []

face_images = [
    (filename, extract_face(os.path.join(image_folder, filename)))
    for filename in os.listdir(image_folder)
    if filename.endswith((".png", ".jpg", ".jpeg", ".gif")) 
    and os.path.isfile(os.path.join(image_folder, filename))
    # Make sure the file is a regular file, not a directory or a symlink
    # This can help avoid issues with broken or invalid file pathsi
    and os.path.getsize(os.path.join(image_folder, filename)) > 0
    # Check if the file is not empty
    # This can help avoid issues with corrupted or invalid images
    # Adjust or remove this condition as per your requirements
]

for filename in os.listdir(image_folder):
    if filename.endswith((".png", ".jpg", ".jpeg", ".gif")):
        file_path = os.path.join(image_folder, filename)
        face = extract_face(file_path)
        if face is not None:
            face_images.append((filename, face))
            # Save the extracted face as a new image
            face_image = Image.fromarray(face)
            face_image.save("faces_mtcnn_ref/" + "face_" + filename)

# Print the list of face images
for name, face in face_images:
    print(name)

end_time = time.time()
execution_time = end_time - start_time
print("Total time take: ", execution_time)
