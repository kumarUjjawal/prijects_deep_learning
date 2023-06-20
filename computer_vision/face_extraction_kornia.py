import os
import cv2
import torch
import kornia as K

def detect_and_save_faces(image_folder, output_folder):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Create the face detector
    face_detection = FaceDetector().to(device, dtype)

    # Load and process each image in the folder
    image_files = os.listdir(image_folder)
    for image_file in image_files:
        # Load the image
        image_path = os.path.join(image_folder, image_file)
        img = K.io.load_image(image_path, K.io.ImageLoadType.RGB8, device=device)[None, ...].to(dtype=dtype)  # BxCxHxW
        img_vis = K.tensor_to_image(img.byte())  # To visualize

        # Perform face detection
        with torch.no_grad():
            dets = face_detection(img)

        # Decode the detections
        dets = [FaceDetectorResult(o) for o in dets]

        # Save the faces with bounding boxes
        for b in dets:
            top_left = b.top_left.int().tolist()
            bottom_right = b.bottom_right.int().tolist()
            scores = b.score.tolist()

            for score, tp, br in zip(scores, top_left, bottom_right):
                x1, y1 = tp
                x2, y2 = br

                if score < 0.7:
                    continue  # Skip detection with low score

                # Draw bounding box on the visualization image
                img_vis = cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Save the face region as a separate image
                face_image = img[:, :, y1:y2, x1:x2]  # Crop the face region
                face_image = K.tensor_to_image(face_image.byte())  # Convert back to image format

                # Convert the face image from RGB to BGR
                face_image_bgr = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)

                # Generate a unique filename for the face image based on coordinates
                face_filename = f"face_{x1}_{y1}_{x2}_{y2}.jpg"
                output_path = os.path.join(output_folder, face_filename)
                cv2.imwrite(output_path, face_image_bgr)

        # Save the visualization image with bounding boxes
        output_vis_path = os.path.join(output_folder, f"detected_faces_{image_file}")
        cv2.imwrite(output_vis_path, img_vis)

# Example usage
detect_and_save_faces("images", "faces")

