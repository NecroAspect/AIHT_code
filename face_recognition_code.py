import face_recognition
from PIL import Image, ImageDraw
import cv2

def capture_image():
    # Open the camera
    cap = cv2.VideoCapture(0)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Display the captured frame
        cv2.imshow('Camera', frame)

        # Check if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # Save the captured image with a custom name
            img_path = f"testpic.jpg"
            cv2.imwrite(img_path, frame)
            print(f"Image saved as: {img_path}")
            break

    # Release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()

# Call the function to capture the image
capture_image()

# Load the known images with their respective names
known_image1 = face_recognition.load_image_file("ayush.jpg")
known_image2 = face_recognition.load_image_file("kanishk.jpg")
known_image3 = face_recognition.load_image_file("bhavya.jpg")

# Encode the known faces
known_face_encoding1 = face_recognition.face_encodings(known_image1)[0]
known_face_encoding2 = face_recognition.face_encodings(known_image2)[0]
known_face_encoding3 = face_recognition.face_encodings(known_image3)[0]

# Create arrays of known face encodings and their corresponding names
known_face_encodings = [
    known_face_encoding1,
    known_face_encoding2,
    known_face_encoding3,
]
known_face_names = [
    "Ayush",
    "kanishk",
    "Bhavya"
]

# Load the unknown image
unknown_image = face_recognition.load_image_file("testpic.jpg")

# Find all the face locations and encodings in the unknown image
face_locations = face_recognition.face_locations(unknown_image)
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

# Convert the image to a PIL Image so we can draw on it
pil_image = Image.fromarray(unknown_image)
draw = ImageDraw.Draw(pil_image)

# Loop through each face found in the unknown image
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    # See if the face matches any known face
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

    name = "Unknown"

    # If a match is found, use the known face name
    if True in matches:
        first_match_index = matches.index(True)
        name = known_face_names[first_match_index]
        print("Found a matching face to unlock") #replace with lock signal

    # Draw a box around the face and label it
    draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))
    text_width, text_height = draw.textsize(name)
    draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
    draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

# Display the image with faces recognized
pil_image.show()