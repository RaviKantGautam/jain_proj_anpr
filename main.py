import streamlit as st
import logging
import imutils
import cv2
import numpy as np
import easyocr
import csv
import os
import warnings
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
import glob

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S',
                    filename='app.log',)
logger.info("Starting the app...")

authentication_status = st.session_state.get("authentication_status", False)


class Authenticator:
    def __init__(self, config_path):
        with open(config_path) as file:
            self.config = yaml.load(file, Loader=SafeLoader)
        self.authenticator = stauth.Authenticate(
            self.config['credentials'],
            self.config['cookie']['name'],
            self.config['cookie']['key'],
            self.config['cookie']['expiry_days'],
        )
        self.authentication_status = st.session_state.get(
            "authentication_status", False)

    def login(self):
        if not self.authentication_status:
            self.authenticator.login(
                location='main',
                max_login_attempts=1,
            )
            self.authentication_status = st.session_state.authentication_status

            if self.authentication_status == False:
                st.error('Username/password is incorrect')
            if self.authentication_status == None:
                st.warning('Please enter your username and password')

    def logout(self):
        if self.authentication_status:
            self.authenticator.logout('Logout', 'main')


class LicensePlateDetector:

    def __init__(self):
        self.reader = easyocr.Reader(['en'])
        self.media_dir = "media"
        if not os.path.exists(self.media_dir):
            logger.warning(
                "Media directory does not exist. Creating a new one...")
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
        keypoints = cv2.findContours(
            edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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
        keypoints = cv2.findContours(
            edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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

authenticator = Authenticator('.streamlit/credential.yml')

if not authentication_status:
    with st.spinner("Authenticating..."):
        try:
            authenticator.logout()
        except:
            pass
        finally:
            authenticator.login()

if authentication_status:
    with st.sidebar:
        authenticator.logout()

    # Create a dropdown to choose between image or video
    option = st.selectbox(
        'Choose the type of file to upload',
        ('Image', 'Video')
    )

    anpr = LicensePlateDetector()

    if option == 'Image':
        # Create a file uploader
        uploaded_file = st.file_uploader("Choose a file", type=[
                                         "jpg", "jpeg", "png"], 
                                         accept_multiple_files=False, 
                                         key="image", help="Upload an image file")

        if uploaded_file:
            # Display the image
            # Loop through the images in the specified directory
            image_files = glob.glob(r'C:\Users\ravikantg\Documents\jain_proj_anpr\media\archive\images\*.jpg') + \
                          glob.glob(r'C:\Users\ravikantg\Documents\jain_proj_anpr\media\archive\images\*.jpeg') + \
                          glob.glob(r'C:\Users\ravikantg\Documents\jain_proj_anpr\media\archive\images\*.png')

            for image_file in image_files:
                st.image(image_file, caption=os.path.basename(image_file), use_column_width=True)
                with st.spinner(f"Processing the image {os.path.basename(image_file)}..."):
                    try:
                        text = anpr.process_image(image_file)
                        st.success(f"Detected license plate number: {text}")
                        # write_to_csv = st.checkbox(f"Write {os.path.basename(image_file)} to CSV", key=image_file)
                        if text:
                            anpr.write_to_csv(text)
                            st.success(f"License plate number {text} has been written to {anpr.csv_file_path}")
                    except Exception as e:
                        st.error(f"An error occurred while processing {os.path.basename(image_file)}: {e}")
            st.image(uploaded_file, caption="Uploaded Image",
                     use_column_width=True)
            file_path = anpr.save_file(uploaded_file)

            with st.spinner("Processing the image..."):
                try:
                    text = anpr.process_image(file_path)
                    st.success(f"Detected license plate number: {text}")
                    write_to_csv = st.checkbox("Write to CSV")
                    if write_to_csv:
                        anpr.write_to_csv(text)
                        st.success(
                            f"License plate number {text} has been written to {anpr.csv_file_path}")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
    else:
        # Create a file uploader
        uploaded_file = st.file_uploader("Choose a file", type=[
                                         "mp4", "avi", "mov"], 
                                         accept_multiple_files=False, 
                                         key="video", help="Upload a video file")

        if uploaded_file:
            # Display the video
            st.video(uploaded_file, start_time=0,
                     autoplay=True, loop=True)

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
                        st.success(
                            f"Detected license plates have been written to {csv_file_path}")
                else:
                    st.info("No license plates detected in the video")
