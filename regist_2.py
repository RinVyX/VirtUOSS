import cv2
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import math
import scipy.optimize as spo
from scipy.optimize import minimize
import time


## --------------------------------- Global variables --------------------------------- ##


reference_points = []; target_points = []; transformed_points = []; target_points_centered = []; reference_points_centered = []
crop_points = []
scale_percent = 40
height = None; width = None; dim = None
resized_image = None; image = None

# Variables for the cropping functions
points = []

## --------------------------------- Registration functions --------------------------------- ##
def calculate_centroid(points):
    return np.mean(points, axis=0)

def transform_points_centered(points, angle, scale_x, scale_y, skew_x, skew_y, stretch_x, stretch_y, translate_x, translate_y, centroid):
    """
    :param points: Nx2 array of points [[x1, y1], [x2, y2], ...]
    :param centroid: The new origin [cx, cy]
    :return: Transformed points
    """
    global transformed_points
    translated_points = points - centroid
    
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    scaling_matrix = np.array([
        [scale_x, 0],
        [0, scale_y]
    ])
    skew_matrix = np.array([
        [1, skew_x],
        [skew_y, 1]
    ])
    stretch_matrix = np.array([
        [stretch_x, 0],
        [0, stretch_y]
    ])
    transformation_matrix = rotation_matrix @ scaling_matrix @ skew_matrix @ stretch_matrix
    transformed_points = translated_points @ transformation_matrix.T
    
    transformed_points += centroid
    
    transformed_points += np.array([translate_x, translate_y])
    
    return transformed_points


def get_affine_matrix_centered(angle, scale_x, scale_y, skew_x, skew_y, stretch_x, stretch_y, translate_x, translate_y, centroid):
    
    translation_to_origin = np.array([
        [1, 0, -centroid[0]],
        [0, 1, -centroid[1]],
        [0, 0, 1]
    ])
    
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    
    scaling_matrix = np.array([
        [scale_x, 0, 0],
        [0, scale_y, 0],
        [0, 0, 1]
    ])
    
    skew_matrix = np.array([
        [1, skew_x, 0],
        [skew_y, 1, 0],
        [0, 0, 1]
    ])
    
    stretch_matrix = np.array([
        [stretch_x, 0, 0],
        [0, stretch_y, 0],
        [0, 0, 1]
    ])
    
    translation_from_origin = np.array([
        [1, 0, centroid[0] + translate_x],
        [0, 1, centroid[1] + translate_y],
        [0, 0, 1]
    ])
    
    transformation_matrix = translation_from_origin @ rotation_matrix @ scaling_matrix @ skew_matrix @ stretch_matrix @ translation_to_origin
    
    affine_matrix = transformation_matrix[:2, :]
    return affine_matrix

def objective_function(params, reference_points, target_points, centroid):
    """
    Calculate the sum of squared distances between transformed points and reference points.
    """
    angle, scale_x, scale_y, skew_x, skew_y, stretch_x, stretch_y, translate_x, translate_y = params
    transformed_points = transform_points_centered(target_points, angle, scale_x, scale_y, skew_x, skew_y, stretch_x, stretch_y, translate_x, translate_y, centroid)
    distances = np.linalg.norm(transformed_points - reference_points, axis=1)
    return np.sum(distances**2)

def transform_image(image, affine_matrix):
    
    height, width = image.shape[:2]
    
    transformed_image = cv2.warpAffine(image, affine_matrix, (width - 500, height + 600), flags=cv2.INTER_LINEAR)
    return transformed_image


def compute_metrics(reference, transformed, name, params, exec_time):
    diffs = transformed - reference
    distances = np.linalg.norm(diffs, axis=1)
    
    return {
        'Name': name,
        'Mean Error': np.mean(distances),
        'RMSE': np.sqrt(np.mean(distances**2)),
        'MAE': np.mean(np.abs(diffs)),
        'Max Error': np.max(distances),
        'Execution Time (s)': exec_time,
        'Parameters': params
    }


## --------------------------------- Main code --------------------------------- ##

if __name__ == "__main__":
    reference_path = "C:\\Users\\P14s\\OneDrive - UQAM\\UQAM\\videos\\Fa2023EnvEnr_EmoANT_IN_21SEP2023_IW2_8554Versace.mp4"
    if reference_path:
        cap = cv2.VideoCapture(reference_path)
        success, reference_image = cap.read()
        reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)
        reference_points = np.array([[275, 317],[977, 335],[935, 1277],[245,1190]], dtype=np.float32)

    target_path = "C:\\Users\\P14s\\OneDrive - UQAM\\UQAM\\videos\\Splits\\Babelle\\Head Shaking (as if to detach)_video.mp4"
    target_path = "C:\\Users\\P14s\\OneDrive - UQAM\\UQAM\\videos\\Splits\\Maisie\\Exploration_video.mp4"
    target_path = "C:\\Users\\P14s\\OneDrive - UQAM\\UQAM\\videos\\Splits\\Mack\\Defecation_video.mp4"
    #target_path = "C:\\Users\\P14s\\OneDrive - UQAM\\UQAM\\videos\\Splits\\Haagendaz\\Scratching_video.mp4"
    if target_path:
        cap = cv2.VideoCapture(target_path)
        success, target_image = cap.read()
        target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
        target_points = np.array([[472.2500, 293.7500],[775.2500, 290.7500],[778.2500, 683.7500],[511.2500, 655.2500]], dtype=np.float32)
        target_points = np.array([[221, 419],[884, 398],[872, 1250],[227, 1241]], dtype=np.float32)
        target_points = np.array([[551.7500, 310.2500],[847.2500, 307.2500],[842.7500, 682.2500],[586.2500, 668.7500]], dtype=np.float32)
        #target_points = np.array([[203, 404],[908, 401],[896, 1262],[212, 1256]], dtype=np.float32)
    ## --------------------------------- Registration --------------------------------- ##

    initial_params = [0, 1, 1, 0, 0, 1, 1, 0, 0]

    centroid = calculate_centroid(target_points)

    progress = []

    def callback(xk):
        progress.append(objective_function(xk, reference_points, target_points, centroid))

    start_time = time.time()

    result = minimize(
        objective_function,
        initial_params,
        args=(reference_points, target_points, centroid),
        method='BFGS',  #L-BFGS-B
        callback=callback
    )
    execution_time = time.time() - start_time
    best_params = result.x
    metrics1 = compute_metrics(reference_points, transformed_points, "BFGS", best_params, execution_time)

    print("Best parameters:", best_params)

    affine_matrix = get_affine_matrix_centered(*best_params, centroid)

    transformed_image = transform_image(target_image, affine_matrix)

    print("transformed points :", transformed_points - calculate_centroid(transformed_points))

    plt.subplot(2, 3, 1)
    plt.imshow(reference_image)
    plt.scatter(np.array(reference_points)[:, 0], np.array(reference_points)[:, 1], c='blue', label='Reference Points')
    plt.plot(np.vstack((reference_points, np.array([reference_points[0, 0], reference_points[0, 1]])))[:, 0], 
             np.vstack((reference_points, np.array([reference_points[0, 0], reference_points[0, 1]])))[:, 1],
             c='blue')
    plt.title('Reference Image')
    
    plt.subplot(2, 3, 2)
    plt.imshow(target_image)
    plt.scatter(np.array(target_points)[:, 0], np.array(target_points)[:, 1], c='red', label='Target Points')
    plt.plot(np.vstack((target_points, np.array([target_points[0, 0], target_points[0, 1]])))[:, 0], 
             np.vstack((target_points, np.array([target_points[0, 0], target_points[0, 1]])))[:, 1],
             c='red')
    plt.title('Target Image')

    plt.subplot(2, 3, 3)
    plt.imshow(transformed_image)
    plt.scatter(np.array(transformed_points)[:, 0], np.array(transformed_points)[:, 1], c='red', label='Transformed Points')
    plt.plot(np.vstack((transformed_points, np.array([transformed_points[0, 0], transformed_points[0, 1]])))[:, 0], 
             np.vstack((transformed_points, np.array([transformed_points[0, 0], transformed_points[0, 1]])))[:, 1],
             c='red')
    plt.scatter(np.array(reference_points)[:, 0], np.array(reference_points)[:, 1], c='blue', label='Reference Points')
    plt.plot(np.vstack((reference_points, np.array([reference_points[0, 0], reference_points[0, 1]])))[:, 0], 
             np.vstack((reference_points, np.array([reference_points[0, 0], reference_points[0, 1]])))[:, 1],
             c='blue', alpha = 0.5)
    plt.title('Transformed Image')

    plt.subplot(2, 3, 4)
    plt.scatter(np.array(reference_points)[:, 0], np.array(reference_points)[:, 1], c='blue', label='Reference Points')
    plt.scatter(np.array(target_points)[:, 0], np.array(target_points)[:, 1], c='red', label='Target Points')

    plt.subplot(2, 3, 5)
    plt.scatter(np.array(reference_points)[:, 0], np.array(reference_points)[:, 1], c='blue', label='Reference Points')
    plt.scatter(np.array(transformed_points)[:, 0], np.array(transformed_points)[:, 1], c='red', label='Transformed Points')

    plt.subplot(2, 3, 6)
    plt.plot(progress, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Sum of Squared Distances')
    plt.title('Optimization Progress')
    plt.grid(True)
    
    #plt.tight_layout()
    plt.show()



# transformation params = [theta, scale_x, scale_y, skew_x, skew_y, tx, ty]
# Babelle transformation = [-3.91852106e-01  2.22792593e+00  2.13253089e+00 -5.05555275e-01  4.70898867e-01  1.11438098e-06  1.90897302e-06]
# Maisie transformation = [-7.54018542e-02  1.05398487e+00  1.07892788e+00 -1.14544821e-01  1.57295191e-01  1.55445728e-06 -4.42890916e-06]
# Mack transformation = [-4.55987511e-01  2.19587658e+00  2.13024819e+00 -5.76289857e-01  5.78169125e-01  3.91231041e-07  1.67374662e-06]
# Haagendaz transformation = [-6.22241161e-03  1.00167322e+00  1.05958916e+00 -4.67632588e-02  7.44400105e-02  7.79304847e-06 -1.79337542e-05]]

# theta = [-0.392, -0.075, -0.456, -0.006] == [0:-1] == [5:-6]
# scale_x = [2.228, 1.054, 2.196, 1.002] == [0:3] == [5:-5]
# scale_y = [2.133, 1.079, 2.130, 1.060] == [0:3] == [5:-5]
# skew_x = [-0.506, -0.115, -0.576, -0.047] == [0:-1] == [5:-6]
# skew_y = [0.471, 0.157, 0.578, 0.074] == [0:1] == [5:-6]
# tx = [1.114, 1.554, 3.912, 7.793] == [1:8] == [6:0]
# ty = [1.909, -4.429, 1.674, -1.793] == [2:-5] == [7:-10]
