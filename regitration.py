import cv2
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import scipy.optimize as spo
from scipy.optimize import minimize


""" 
- This script is used to register two images using a set of corresponding points:
- points are selected by the user in the reference and target images
- An image crop is added to the script to target the are of study (FUNCTION TO ADD!!!!!!!!)

Functions:
- load_image(image_type): Opens a file dialog to load an image.
- get_points(event, x, y, flags, param): Mouse callback function to get points from the image.
- reset_callback(image_type): Resets the points and callback for the specified image type.
- process_image(image_path, image_type): Processes the image and gets points from the user.
- crop_image(image, points): Crops the image based on the chosen points.
- translate_points_origin(points): Translates points so that the center of the group of points is at the origin.
- translate_image_origin(image, points): Translates the image so that the center of the points is at the origin.
- my_dist(x, y): Calculates the distance between two sets of points.
- translation_pts(current, w): Applies translation to the points.
- rotation_pts(current, angle): Applies rotation to the points.
- scale_x_pts(current, scale): Applies scaling in the x direction to the points.
- scale_y_pts(current, scale): Applies scaling in the y direction to the points.
- skew_x_pts(current, skew): Applies skewing in the x direction to the points.
- skew_y_pts(current, skew): Applies skewing in the y direction to the points.
- objective(params): Objective function to minimize the sum of distances between transformed points and target points.
- apply_perspective_transform(image, result, src_points): Applies the perspective transformation to the image.


Main code:
- Loads the reference and target images.
- Processes the images to get the points.
- Crops the images based on the selected points. <== Replace with the function to crop the image (user input)
- Translates the images so that the center of the points is at the origin.
- Optimizes the transformation parameters.
- Applies the transformation to the target image.
- Displays the results.

To come:
- Reads a CSV file and extracts bounding box coordinates and behaviors. 

"""


## --------------------------------- Global variables --------------------------------- ##

# Initialize lists to store the points
reference_points = []; target_points = []; transformed_points = []
target_points_centered = []; reference_points_centered = []; crop_points_centered = []
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
    elif image_type == "crop_image":
        global crop_points
        crop_points = []
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

## --------------------------------- Registration functions --------------------------------- ##

# Calculate the center of the points
def calculate_center(points):
    return np.mean(points, axis=0)

# Translate the points so that the center is at the origin
def translate_points_origin(points):
    center = calculate_center(points)
    return points - center

# Define the affine transformation matrix
def affine_transform(params, points):
    theta, scale_x, scale_y, skew_x, skew_y, tx, ty = params
    # Rotation matrix
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    # Scaling matrix
    scaling_matrix = np.array([
        [scale_x, 0],
        [0, scale_y]
    ])
    # Skewing matrix
    skewing_matrix = np.array([
        [1, skew_x],
        [skew_y, 1]
    ])
    # Combine transformation matrix
    transformation_matrix = rotation_matrix @ scaling_matrix @ skewing_matrix
    # Apply transformation
    transformed_points = points @ transformation_matrix.T + np.array([tx, ty])
    return transformed_points

# Objective function to minimize
def objective_function(params):
    transformed_points = affine_transform(params, target_points_centered)
    return np.sum((transformed_points - reference_points_centered) ** 2)

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
    print('transformation_matrix:\n',transformation_matrix)
    return transformation_matrix[:2, :]


## --------------------------------- Main code --------------------------------- ##

if __name__ == "__main__":

    # Load the reference image & target image
    reference_path = load_image("reference")
    if reference_path:
        reference_points = process_image(reference_path, "reference")
        if reference_points is None:
            exit()  # Exit if there was an error loading the reference image
        
        reference_points = [[275, 317], [977, 335], [935, 1277], [245, 1190]]
        cap = cv2.VideoCapture(reference_path)
        success, reference_image = cap.read()
        reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)

        """ plt.subplot(2, 2, 1)
        plt.imshow(reference_image)
        # Plot the points in a separate plot
        plt.subplot(2, 2, 2)
        plt.title("Reference Points")
        plt.xlim(0, int(reference_image.shape[1]))  # Set x-axis limits to original image width
        plt.ylim(int(reference_image.shape[0]), 0)   # Set y-axis limits to original image height (inverted)
        plt.gca().set_aspect('equal', adjustable='box')  # Keep aspect ratio

        # Extract and adjust points for plotting
        if reference_points:
            x_points, y_points = zip(*reference_points)
            plt.scatter(x_points, y_points, color='red', s=50)

        plt.xlabel("Width")
        plt.ylabel("Height") """

    target_path = load_image("target")
    if target_path:
        target_points = process_image(target_path, "target")
        if target_points is None:
            exit()  # Exit if there was an error loading the target image
        
        target_points = [[472.2500, 293.7500], [775.2500, 290.7500], [778.2500, 683.7500], [511.2500, 655.2500]]
        cap = cv2.VideoCapture(target_path)
        success, target_image = cap.read()
        target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
        
        """ plt.subplot(2, 2, 3)
        plt.imshow(target_image)
        #plt.imshow(target_image[(int(np.min(np.array(target_points)[:, 1])) - 200):(int(np.max(np.array(target_points)[:, 1])) + 200), (int(np.min(np.array(target_points)[:, 0])) - 200):(int(np.max(np.array(target_points)[:, 0])) + 200)])
        
        # Plot the points in a separate plot
        plt.subplot(2, 2, 4)
        plt.title("Target Points")
        plt.xlim(0, int(target_image.shape[1]))  # Set x-axis limits to original image width
        plt.ylim(int(target_image.shape[0]), 0)   # Set y-axis limits to original image height (inverted)
        plt.gca().set_aspect('equal', adjustable='box')  # Keep aspect ratio

        # Extract and adjust points for plotting
        if target_points:
            x_points, y_points = zip(*target_points)
            plt.scatter(x_points, y_points, color='blue', s=50)

        plt.xlabel("Width")
        plt.ylabel("Height")

    # Show the plot
    plt.tight_layout()
    plt.show() """

    # Crop the images based on the selected points
    #crop_path = load_image("target to crop")
    if target_path:
        crop_points = process_image(target_path, "crop_image")
        if crop_points is None:
            exit()  # Exit if there was an error loading the target image

        # crop points as 4 points
        cropped_target_image = crop_image(target_image, crop_points)
        """ plt.subplot(1, 2, 1)
        plt.imshow(target_image)
        plt.title("Original Target Image")
        plt.subplot(1, 2, 2)
        plt.imshow(cropped_target_image)
        plt.title("Cropped Target Image")
        plt.show()    """     
        top_left = crop_points[0]
        bottom_right = crop_points[1]
        crop_points = [[top_left[0], top_left[1]], [bottom_right[0], top_left[1]], [bottom_right[0], bottom_right[1]], [top_left[0], bottom_right[1]]]
        print('crop_points:\n',crop_points)
        print('reference_points:\n',reference_points)
    ## --------------------------------- Registration --------------------------------- ##

    # Initial guess for the transformation parameters
    initial_params = [0, 1, 1, 0, 0, 0, 0]  # [theta, scale_x, scale_y, skew_x, skew_y, tx, ty]

    target_center = calculate_center(target_points)
    reference_center = calculate_center(reference_points)
    crop_center = calculate_center(crop_points)

    # translate the image so that the center of the points is at the origin
    target_points_centered = translate_points_origin(target_points)
    reference_points_centered = translate_points_origin(reference_points)
    crop_points_centered = translate_points_origin(crop_points)

    print('target_points_centered:\n',target_points_centered)
    print('reference_points_centered:\n',reference_points_centered)
    print('crop_points_centered:\n',crop_points_centered)
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
        [1, 0, crop_center[0]],
        [0, 1, crop_center[1]],
        [0, 0, 1]
    ])

    # Combine the transformations
    combined_matrix = translation_matrix_from_origin @ np.vstack([affine_matrix, [0, 0, 1]]) @ translation_matrix_to_origin

    crop_points = np.array(crop_points)
    reference_points = np.array(reference_points)

    # Calculate the scaling factors
    scale_x = reference_points[1, 0] - reference_points[0, 0] / crop_points[1, 0] - crop_points[0, 0] # reference_width / crop_width
    scale_y = reference_points[2, 1] - reference_points[1, 1] / crop_points[2, 1] - crop_points[1, 1] # reference_height / crop_height

    # Scale the crop_points
    scaled_crop_points = crop_points * [scale_x, scale_y]
    print('scaled_crop_points:\n',scaled_crop_points)
    # Calculate the width and height of the crop
    width = max([point[0] for point in scaled_crop_points]) - min([point[0] for point in scaled_crop_points])
    height = max([point[1] for point in scaled_crop_points]) - min([point[1] for point in scaled_crop_points])
    
    # Apply the combined transformation to the target image
    transformed_target_image = cv2.warpAffine(target_image, combined_matrix[:2, :], (reference_image.shape[1], reference_image.shape[0]))

    # Process the entire video
    cap = cv2.VideoCapture(target_path)
    """ frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) 
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) """
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter('C:\\Users\\P14s\\OneDrive - UQAM\\UQAM\\videos\\normalized\\output_video_test.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 
                          fps, (reference_image.shape[1], reference_image.shape[0]))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        transformed_frame = cv2.warpAffine(frame, combined_matrix[:2, :], (reference_image.shape[1], reference_image.shape[0]))
        transformed_frame = cv2.cvtColor(transformed_frame, cv2.COLOR_RGB2BGR)
        out.write(transformed_frame)

    cap.release()
    out.release()

    # Apply the optimized transformation to the target points
    transformed_target_points_centered = affine_transform(optimized_params, target_points_centered)
    print('transformed_target_points_centered:\n',transformed_target_points_centered)
    transformed_target_points = transformed_target_points_centered + reference_center

    # Plot the results
    plt.figure(figsize=(15, 10))

    # Plot reference image
    plt.subplot(2, 3, 1)
    plt.imshow(target_image)
    plt.plot(np.array(target_points), c='green')
    plt.scatter(np.array(target_points)[:, 0], np.array(target_points)[:, 1], c='green')
    plt.title('Target Image')

    plt.subplot(2, 3, 2)
    plt.imshow(reference_image)
    plt.plot(np.array(reference_points), c='red')
    plt.scatter(np.array(reference_points)[:, 0], np.array(reference_points)[:, 1], c='red')
    plt.title('Reference Image')

    plt.subplot(2, 3, 3)
    plt.scatter(np.array(reference_points_centered)[:, 0], np.array(reference_points_centered)[:, 1], c='red')
    plt.scatter(np.array(target_points_centered)[:, 0], np.array(target_points_centered)[:, 1], c='green')
    plt.title('Stall corners before registration')

    # Plot transformed target image
    plt.subplot(2, 3, 4)
    # show only the region of interest of the target image
    plt.imshow(target_image[(int(np.min(np.array(target_points)[:, 1])) - 200):(int(np.max(np.array(target_points)[:, 1])) + 200), (int(np.min(np.array(target_points)[:, 0])) - 200):(int(np.max(np.array(target_points)[:, 0])) + 200)])
    plt.title('Target Image')

    plt.subplot(2, 3, 5)
    plt.imshow(transformed_target_image)
    plt.plot(np.array(reference_points))
    plt.scatter(np.array(reference_points)[:, 0], np.array(reference_points)[:, 1], c='red')
    plt.title('Transformed Target Image')

    plt.subplot(2, 3, 6)
    plt.scatter(np.array(reference_points)[:, 0], np.array(reference_points)[:, 1], c='red', label='Reference Points')
    plt.scatter(np.array(transformed_target_points)[:, 0], np.array(transformed_target_points)[:, 1], c='green', label='Transformed Target Points')
    plt.title('Stall corners after registration')

    plt.legend()
    plt.show()