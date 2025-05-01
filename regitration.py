import cv2
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import scipy.optimize as spo
from scipy.optimize import minimize

## --------------------------------- Global variables --------------------------------- ##

# Initialize lists to store the points
reference_points = []; target_points = []; transformed_points = []; target_points_centered = []; reference_points_centered = []
crop_points = []
scale_percent = 40  # percent of original size
height = None; width = None; dim = None # Initialize height, width & dim
resized_image = None; image = None  # Initialize image & resized_image

# Variables for the cropping functions
points = []
## --------------------------------- Define functions --------------------------------- ##

# Function to open file dialog and load the image
def load_image(image_type):
    root = tk.Tk()
    root.withdraw()  # Hide the root window!!!
    file_path = filedialog.askopenfilename(title=f"Select the {image_type} Image", filetypes=[("Selected Files", "*.mp4;*.avi")])
    return file_path

# the mouse callback func
def get_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Append the point to the appropriate list
        if param == "reference":
            reference_points.append((x, y))
        elif param == "target":
            target_points.append((x, y))        
        # Draw a circle at the clicked point
        cv2.circle(image, (x, y), 10, (0, 255, 0), -1)
        # Display the coordinates on the image
        # here we display the x and y after scaling them to the original image size, so it shows the real values
        cv2.putText(image, f'({x}, {y})', (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        # display image with the drwan point
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Image", image)

# Mouse callback function to capture points
def get_points_crop(event, x, y, flags, param):
    global crop_points, image
    if event == cv2.EVENT_LBUTTONDOWN:
        # Append the point to the appropriate list
        crop_points.append((x, y))
        if len(crop_points) == 2:  # Two points define a rectangle
            cv2.rectangle(image, crop_points[0], crop_points[1], (0, 255, 0), 2)
            cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
            cv2.imshow("Image", image)


# Function to reset the points and callback
def reset_callback(image_type):
    # making the reference_points & target_points global to use afterwards outside the function
    if image_type == "reference":
        global reference_points
        reference_points = []
    elif image_type == "target":
        global target_points
        target_points = []
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Image", image)  # Re-display the image
    cv2.setMouseCallback("Image", get_points, param=image_type)  # Reset the callback

# Function to process the image and get points
def process_image(image_path, image_type):
    global image  
    # capturing the first frame to use as a reference
    cap = cv2.VideoCapture(image_path)
    if not cap.isOpened(): # Error handling
        print(f"Error: Could not open video file {image_path}")
        return None
    
    success, image = cap.read()
    if not success: # Error handling
        print("Error: Could not read image from video.")
        return None
    
    # Display the image and set the mouse callback function
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Image", image)
    if image_type == "reference":
        cv2.setMouseCallback("Image", get_points, param=image_type)
        while True:
            key = cv2.waitKey(1)
            if len(reference_points) >= 4:
                break
            elif key == ord('r'):
                reset_callback(image_type)

    elif image_type == "target":
        cv2.setMouseCallback("Image", get_points, param=image_type)
        # Wait until 4 points are clicked or reset
        while True:
            key = cv2.waitKey(1)
            if len(target_points) >= 4:
                break
            elif key == ord('r'):
                reset_callback(image_type)

    elif image_type == "crop_image":
        cv2.setMouseCallback("Image", get_points_crop, param=image_type)
        # Wait until lenght is 2 or reset
        while True:
            key = cv2.waitKey(1)
            if len(crop_points) ==  2:
                break
            elif key == ord('r'):
                reset_callback(image_type)
        
    cv2.destroyAllWindows()
    # the returned points here are in their original size (no need to scale them) (got them from the mouse callback)
    if image_type == "reference":
        return reference_points
    elif image_type == "target":
        return target_points
    elif image_type == "crop_image":
        return crop_points


# Function to crop the image using the extracted points
def crop_image(image, points):
    if len(points) != 2:
        raise ValueError("Exactly two points are required to define a rectangle.")

    # Ensure the points are in the correct order (top-left and bottom-right)
    x1, y1 = points[0]
    x2, y2 = points[1]
    top_left = (min(x1, x2), min(y1, y2))
    bottom_right = (max(x1, x2), max(y1, y2))

    # Crop the image
    cropped_image = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    return cropped_image

# Function to pad the image with black pixels to match the desired dimensions
def pad_image_to_size(image, target_width, target_height):
    height, width = image.shape[:2]
    pad_width = max(0, target_width - width)
    pad_height = max(0, target_height - height)
    
    # Calculate padding for left, right, top, and bottom
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    
    # Pad the image with black pixels
    padded_image = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return padded_image

# Function to crop and pad the image to match the dimensions of the cropped target image
def crop_and_pad_image(image, crop_points, target_width, target_height):
    cropped_image = crop_image(image, crop_points)
    padded_image = pad_image_to_size(cropped_image, target_width, target_height)
    return padded_image

## --------------------------------- Registration functions --------------------------------- ##

# Compute the affine transformation matrix for the image
def get_affine_matrix(params, center):
    theta, scale_x, scale_y, skew_x, skew_y, tx, ty = params
    # Rotation matrix
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    # Scaling matrix
    scaling_matrix = np.array([
        [scale_x, 0, 0],
        [0, scale_y, 0],
        [0, 0, 1]
    ])
    # Skewing matrix
    skewing_matrix = np.array([
        [1, skew_x, 0],
        [skew_y, 1, 0],
        [0, 0, 1]
    ])
    # Translation matrix
    translation_matrix = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ])
    # Combined transformation matrix
    transformation_matrix = translation_matrix @ rotation_matrix @ scaling_matrix @ skewing_matrix
    return transformation_matrix[:2, :]


## --------------------------------- Main code --------------------------------- ##

if __name__ == "__main__":

    # Load the reference image & target image
    reference_path = load_image("reference")
    if reference_path:
        reference_points = process_image(reference_path, "reference")
        if reference_points is None:
            exit()  # Exit if there was an error loading the reference image
        
        cap = cv2.VideoCapture(reference_path)
        success, reference_image = cap.read()
        reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)


    target_path = load_image("target")
    if target_path:
        target_points = process_image(target_path, "target")
        if target_points is None:
            exit()  # Exit if there was an error loading the target image
        
        cap = cv2.VideoCapture(target_path)
        success, target_image = cap.read()
        target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)


    # Crop the images based on the selected points
    #crop_path = load_image("target to crop")
    if target_path:
        crop_points = process_image(target_path, "crop_image")
        if crop_points is None:
            exit()  # Exit if there was an error loading the target image

        cropped_target_image = crop_image(target_image, crop_points)
        target_height, target_width = cropped_target_image.shape[:2]

        # Crop and pad the reference image to match the dimensions of the cropped target image
        cropped_reference_image = crop_and_pad_image(reference_image, crop_points, target_width, target_height)
        
        plt.subplot(2, 2, 1)
        plt.imshow(target_image)
        # Extract and adjust points for plotting
        if target_points:
            x_points, y_points = zip(*target_points)
            plt.scatter(x_points, y_points, color='blue', s=50)
        plt.title("Original Target Image")
        plt.subplot(2, 2, 2)
        plt.imshow(cropped_target_image)
        plt.title("Cropped Target Image")
        plt.subplot(2, 2, 3)
        plt.imshow(reference_image)
        # Extract and adjust points for plotting
        if reference_points:
            x_points, y_points = zip(*reference_points)
            plt.scatter(x_points, y_points, color='red', s=50)
        plt.title("Original reference Image")
        plt.subplot(2, 2, 4)
        plt.imshow(cropped_reference_image)
        plt.title("Cropped reference Image")
        plt.tight_layout()
        plt.show()        

    ## --------------------------------- Registration --------------------------------- ##

    # Initial guess for the transformation parameters
    initial_params = [0, 1, 1, 0, 0, 0, 0]  # [theta, scale_x, scale_y, skew_x, skew_y, tx, ty]

    target_center = calculate_center(target_points)
    reference_center = calculate_center(reference_points)

    # translate the image so that the center of the points is at the origin
    target_points_centered = translate_points_origin(target_points)
    reference_points_centered = translate_points_origin(reference_points)

    # minimize the objective function to get the optimized parameters
    result = minimize(objective_function, initial_params, method='BFGS')
    optimized_params = result.x
    print('optimized_params:\n',optimized_params)

    # Get the affine matrix
    affine_matrix = get_affine_matrix(optimized_params, target_center)

    # Translate the target image so that the center of the points is at the origin
    translation_matrix_to_origin = np.array([
        [1, 0, -target_center[0]],
        [0, 1, -target_center[1]],
        [0, 0, 1]
    ])

    # Translate the target image back to the original coordinate system
    translation_matrix_from_origin = np.array([
        [1, 0, reference_center[0]],
        [0, 1, reference_center[1]],
        [0, 0, 1]
    ])

    # Combine the transformations
    combined_matrix = translation_matrix_from_origin @ np.vstack([affine_matrix, [0, 0, 1]]) @ translation_matrix_to_origin

    # Apply the combined transformation to the cropped target image
    transformed_target_image = cv2.warpAffine(cropped_target_image, combined_matrix[:2, :], (target_width, target_height))

    """ # Process the entire video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) 
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap = cv2.VideoCapture(target_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    # get the path from the user to save the video
    # user choose the folder and name of the video
    # Let the user choose the save location and file name
    out_path = filedialog.asksaveasfilename(
        title='Save the video',
        filetypes=[('MP4 files', '*.mp4'), ('AVI files', '*.avi')],
        defaultextension='.mp4'  # Ensure a default extension is added
    )
    # Check if the user provided a valid path
    if not out_path:
        print('Error: No file path selected. Exiting.')
        exit()

    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for AVI files
    out = cv2.VideoWriter(out_path, fourcc, fps, (reference_image.shape[1], reference_image.shape[0]))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        transformed_frame = cv2.warpAffine(frame, combined_matrix[:2, :], (reference_image.shape[1], reference_image.shape[0]))
        transformed_frame = cv2.cvtColor(transformed_frame, cv2.COLOR_RGB2BGR)
        out.write(transformed_frame)

    cap.release()
    out.release() """

    # Apply the optimized transformation to the target points
    transformed_target_points_centered = affine_transform(optimized_params, target_points_centered)
    transformed_target_points = transformed_target_points_centered + reference_center


    # Plot the results
    plt.figure(figsize=(15, 10))

    # Plot reference image
    plt.subplot(1, 2, 1)
    plt.imshow(reference_image)
    plt.title('Reference Image')
    print(reference_points)
    print(np.array(reference_points))
    plt.scatter(np.array(reference_points)[:, 0], np.array(reference_points)[:, 1], c='blue', label='Reference Points')
    for i, (x, y) in enumerate(np.array(reference_points)):
        plt.text(x, y, f'{i+1}', fontsize=12, color='blue')

    # Plot transformed target image
    plt.subplot(1, 2, 2)
    plt.imshow(transformed_target_image)
    plt.imshow(reference_image, alpha=0.5)
    plt.title('Transformed Target Image')
    print(transformed_target_points)
    plt.scatter(np.array(reference_points)[:, 0], np.array(reference_points)[:, 1], c='blue', label='Reference Points')
    for i, (x, y) in enumerate(np.array(reference_points)):
        plt.text(x, y, f'{i+1}', fontsize=12, color='blue')

    plt.scatter(np.array(transformed_target_points)[:, 0], np.array(transformed_target_points)[:, 1], c='red', label='Transformed Target Points')
    for i, (x, y) in enumerate(np.array(transformed_target_points)):
        plt.text(x, y, f'{i+1}', fontsize=12, color='red')

    plt.legend()
    plt.show()