import streamlit as st
import logging
import imutils
import cv2
import numpy as np
import easyocr
import csv
import os
import warnings

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S',
                    filename='app.log',)
logger.info("Starting the app...")

class LicensePlateDetector:
    def __init__(self):
        self.reader = easyocr.Reader(['en'])
        self.media_dir = "media"
        if not os.path.exists(self.media_dir):
            logger.warning("Media directory does not exist. Creating a new one...")
            os.makedirs(self.media_dir)
        self.csv_file_path = 'license_plates.csv'

    def save_file(self, uploaded_file):
        file_path = os.path.join(self.media_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path

    def process_image(self, file_path):
        logger.info(f"Processing image: {file_path}")
        img = cv2.imread(file_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bfilter = cv2.bilateralFilter(gray, 11, 11, 17)
        edged = cv2.Canny(bfilter, 30, 200)
        keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(keypoints)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        location = None
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 10, True)
            if len(approx) == 4:
                location = approx
                break
        if location is None:
            raise ValueError("No license plate found")
        mask = np.zeros(gray.shape, np.uint8)
        new_image = cv2.drawContours(mask, [location], 0, 255, -1)
        new_image = cv2.bitwise_and(img, img, mask=mask)
        (x, y) = np.where(mask == 255)
        (x1, y1) = (np.min(x), np.min(y))
        (x2, y2) = (np.max(x), np.max(y))
        cropped_image = gray[x1:x2+3, y1:y2+3]
        result = self.reader.readtext(cropped_image)
        if not result:
            logger.error("No text found")
            raise ValueError("No text found")
        return result[0][1]

    def process_video(self, file_path):
        logger.info(f"Processing video: {file_path}")
        cap = cv2.VideoCapture(file_path)
        frame_count = 0
        detected_plates = set()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % 30 == 0:
                try:
                    plate = self.process_image_from_frame(frame)
                    detected_plates.add(plate)
                except ValueError:
                    continue
        cap.release()
        return detected_plates

    def process_image_from_frame(self, frame):
        logger.info("Processing image from frame")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bfilter = cv2.bilateralFilter(gray, 11, 11, 17)
        edged = cv2.Canny(bfilter, 30, 200)
        keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(keypoints)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        location = None
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 10, True)
            if len(approx) == 4:
                location = approx
                break
        if location is None:
            raise ValueError("No license plate found")
        mask = np.zeros(gray.shape, np.uint8)
        new_image = cv2.drawContours(mask, [location], 0, 255, -1)
        new_image = cv2.bitwise_and(frame, frame, mask=mask)
        (x, y) = np.where(mask == 255)
        (x1, y1) = (np.min(x), np.min(y))
        (x2, y2) = (np.max(x), np.max(y))
        cropped_image = gray[x1:x2+3, y1:y2+3]
        result = self.reader.readtext(cropped_image)
        if not result:
            logger.error("No text found")
            raise ValueError("No text found")
        return result[0][1]

    def write_to_csv(self, text):
        logger.info(f"Writing text to CSV: {text}")
        with open(self.csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([text])

    def write_multiple_to_csv(self, texts):
        logger.info(f"Writing multiple texts to CSV: {texts}")
        with open(self.csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            for text in texts:
                writer.writerow([text])


# Set the title of the app
st.title("VEHICLE NUMBER PLATE DETECTION AND LOGGING Project")

# Add a subheader
st.subheader("Jain University - Ravi Kant Gautam (221VMTR02192)")

# Create a login form
if 'logged_in' not in st.session_state or not st.session_state['logged_in']:
    st.header("Login")
    with st.form(key='login_form'):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_button = st.form_submit_button("Login")
    if login_button:
        if username == "admin" and password == "admin":
            # Create a session state for the logged-in user
            if 'logged_in' not in st.session_state:
                st.session_state['logged_in'] = True
            st.session_state['username'] = username
            # Save credentials to a file
            logger.info("User logged in successfully")
            st.success("Logged in successfully!")
        else:
            st.error("Invalid username or password")

if st.session_state.get('logged_in'):
    st.write("You are logged in!")

    # Create a dropdown to choose between image or video
    option = st.selectbox(
        'Choose the type of file to upload',
        ('Image', 'Video')
    )

    anpr = LicensePlateDetector()


    if option == 'Image':
        # Create a file uploader
        uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"], accept_multiple_files=False)

        if uploaded_file:
            # Display the image
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

            file_path = anpr.save_file(uploaded_file)

            with st.spinner("Processing the image..."):
                try:
                    text = anpr.process_image(file_path)
                    st.success(f"Detected license plate number: {text}")
                    write_to_csv = st.checkbox("Write to CSV")
                    if write_to_csv:
                        anpr.write_to_csv(text)
                        st.success(f"License plate number {text} has been written to {anpr.csv_file_path}")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.info("Please upload a file")
    else:
        # Create a file uploader
        uploaded_file = st.file_uploader("Choose a file", type=["mp4", "avi", "mov"], accept_multiple_files=False)

        if uploaded_file:
            # Display the video
            st.video(uploaded_file, start_time=0)

            with st.spinner("Processing the video..."):
                file_path = anpr.save_file(uploaded_file)

                detected_plates = anpr.process_video(file_path)

                if detected_plates:
                    csv_file_path = anpr.csv_file_path
                    st.write("Detected license plates:")
                    for plate in detected_plates:
                        st.write(plate)
                    write_to_csv = st.checkbox("Write to CSV")
                    if write_to_csv:
                        anpr.write_multiple_to_csv(detected_plates)
                        st.success(f"Detected license plates have been written to {csv_file_path}")
                else:
                    st.info("No license plates detected in the video")
        else:
            st.info("Please upload a file")
