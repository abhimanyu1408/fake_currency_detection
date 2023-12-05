import cv2
import numpy as np
import streamlit as st

# Define your functions here

def check_color(reference_image, captured_image, threshold=30):
    st.write("Color_check:-")
    st.write("\n")

    fake = 0
    captured_hsv = cv2.cvtColor(captured_image, cv2.COLOR_BGR2HSV)
    reference_hsv = cv2.cvtColor(reference_image, cv2.COLOR_BGR2HSV)

    captured_hsv[:, :, 2] = cv2.equalizeHist(captured_hsv[:, :, 2])
    reference_hsv[:, :, 2] = cv2.equalizeHist(reference_hsv[:, :, 2])

    x, y, width, height = 1240, 0, 150, 800

    captured_hsv = captured_hsv[y:y + height, x:x + width]
    reference_hsv = reference_hsv[y:y + height, x:y + width]

    captured_mean_saturation = np.mean(captured_hsv[:, :, 1])
    reference_mean_saturation = np.mean(reference_hsv[:, :, 1])

    saturation_difference = abs(reference_mean_saturation - captured_mean_saturation)

    if saturation_difference > threshold:
        fake = 1
        st.write("Currency image has different color illumination than the original.")
    else:
        st.write("Currency image has similar color illumination as the original.")
    st.write("------------------------------------------------------------------")

    return fake

def check_bleedlines(reference_left_lines, current_left_lines):
    fake = 0

    # get the difference between the bleedlines of the current image and the bleedlines of the reference image
    d1 = abs(reference_left_lines[0].shape[1] - current_left_lines[0].shape[1])
    d2 = abs(reference_left_lines[1].shape[1] - current_left_lines[1].shape[1])
    d3 = abs(reference_left_lines[2].shape[1] - current_left_lines[2].shape[1])

    for i in range(3):
        reference_image = reference_left_lines[i]
        current_image = current_left_lines[i]

        st.image(cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB), caption=f"Reference Image {i+1}")
        st.image(cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB), caption=f"Current Image {i+1}")

        reference_width = reference_image.shape[1]
        current_width = current_image.shape[1]
        st.write(f"Real blend_line {i+1}: {reference_width} -------- Current blend_line {i+1}: {current_width}")

    if d1 > 2 or d2 > 2 or d3 > 2:
        fake = 1
        st.write("The currency is Fake")
    else:
        st.write("The currency is Real")

    return fake

def extract_features(current_image):
    # Your existing code for feature extraction goes here
    pass

def check_font(current_bottom_serial):
    flag = 0

    # Your existing code for font check goes here

    if flag:
        st.write("Font size check: Fake")
    else:
        st.write("Font size check: Real")

    return flag

def check_serial(reference_bottom_serial, reference_top_serial, current_bottom_serial, current_top_serial, threshold=5):
    fake = 0
    st.write("Serial_check:-")

    # Your existing code for serial check goes here

    if abs(reference_mean_saturation - 81.65454977429319) > threshold:
        fake = 1
        st.write("Serial check: Fake")
    else:
        st.write("Serial check: Real")

    st.write("------------------------------------------------------------------")

    return fake

def check_fake(reference_image_path, current_image_path):
    reference_image_path = cv2.imread(reference_image_path)
    reference_left_lines, reference_bottom_serial, reference_top_serial = extract_features(reference_image_path)

    current_image_path = cv2.imread(current_image_path)
    current_left_lines, current_bottom_serial, current_top_serial = extract_features(current_image_path)

    color = check_color(reference_image_path, current_image_path)
    bleedlines = check_bleedlines(reference_left_lines, current_left_lines)
    serial = check_serial(reference_bottom_serial[0], reference_top_serial[0], current_bottom_serial[0], current_top_serial[0])
    font = check_font(current_bottom_serial[0])

    if color:
        return "The Currency Is Fake --- Reason: Color Intensity"

    if bleedlines:
        return "The Currency Is Fake --- Reason: Bleedlines"

    if serial:
        return "The Currency Is Fake --- Reason: Serial no."

    if font:
        return "The Currency Is Fake --- Reason: Font Size"

    return "The Currency Is Real"

def main():
    st.title("Fake Currency Checker")
    st.subheader("Upload an image")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg"])

    if uploaded_file is not None:
        try:
            # Read the uploaded file
            file_bytes = uploaded_file.read()

            if len(file_bytes) == 0:
                st.write("Error: Empty file uploaded.")
                return

            nparr = np.frombuffer(file_bytes, np.uint8)

            # Decode the image
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                st.write("Error: Invalid or unsupported file format.")
                return

            # Run the fake currency check
            output = check_fake(reference_image_path, image)
            st.write(output)

            # Display the uploaded image
            st.image(image, caption="Uploaded Image", use_column_width=True)

        except Exception as e:
            st.write(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
