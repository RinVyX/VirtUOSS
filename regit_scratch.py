import numpy as np
import matplotlib.pyplot as plt
import cv2

distances = []

def transform_points_combined(points, angle, scale_x, scale_y, skew_x, skew_y, stretch_x, stretch_y, translate_x, translate_y, centroid):
    """
    Apply all transformations (rotation, scaling, skewing, stretching, translation) at once.
    """
    # Translate points to origin
    translated_points = points - centroid
    
    # Rotation matrix
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])    
    # Scaling matrix
    scaling_matrix = np.array([
        [scale_x, 0],
        [0, scale_y]
    ])    
    # Skew matrix
    skew_matrix = np.array([
        [1, skew_x],
        [skew_y, 1]
    ])    
    # Stretch matrix
    stretch_matrix = np.array([
        [stretch_x, 0],
        [0, stretch_y]
    ])
    # Combine transformations
    transformation_matrix = rotation_matrix @ scaling_matrix @ skew_matrix @ stretch_matrix
    # Apply transformation
    transformed_points = translated_points @ transformation_matrix.T    
    # Translate points back to the original coordinate system
    transformed_points += centroid    
    # Apply final translation
    transformed_points += np.array([translate_x, translate_y])    
    return transformed_points

def calculate_distance(reference_points, transformed_points):
    global distances
    distances = np.linalg.norm(transformed_points - reference_points, axis=1)
    return np.sum(distances**2)

# Example reference and target points (replace with your data)
reference_path = "C:\\Users\\P14s\\OneDrive - UQAM\\UQAM\\videos\\Fa2023EnvEnr_EmoANT_IN_21SEP2023_IW2_8554Versace.mp4"
target_path = "C:\\Users\\P14s\\OneDrive - UQAM\\UQAM\\videos\\Splits\\Mack\\Defecation_video.mp4"
if reference_path:
    cap = cv2.VideoCapture(reference_path)
    success, reference_image = cap.read()
    reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)
    reference_points = np.array([[275, 317],[977, 335],[935, 1277],[245,1190]], dtype=np.float32)
if target_path:
    cap = cv2.VideoCapture(target_path)
    success, target_image = cap.read()
    target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
    target_points = np.array([[551.7500, 310.2500],[847.2500, 307.2500],[842.7500, 682.2500],[586.2500, 668.7500]], dtype=np.float32)

# Calculate centroid
centroid = np.mean(target_points, axis=0)

# Init best values
best_distance = float('inf')
best_params = None

# Define parameter ranges
angles = np.linspace(0, 2 * np.pi, 36) # theta = [-0.392, -0.075, -0.456, -0.006] == [0:-1] == [5:-6]
scale_values = np.linspace(-5, 5, 5) # scale_x = [2.228, 1.054, 2.196, 1.002] == [0:3] == [5:-5]
skew_values = np.linspace(-5, 5, 5) # skew_x = [-0.506, -0.115, -0.576, -0.047] == [0:-1] == [5:-6]
stretch_values = np.linspace(-10, 10, 10) # 
translate_values = np.linspace(-10, 10, 5) # ty = [1.909, -4.429, 1.674, -1.793] == [2:-5] == [7:-10]

# Iterate over all combinations of parameters
for angle in angles:
    for scale_x in scale_values:
        for scale_y in scale_values:
            for skew_x in skew_values:
                for skew_y in skew_values:
                    for stretch_x in stretch_values:
                        for stretch_y in stretch_values:
                            for translate_x in translate_values:
                                for translate_y in translate_values:
                                    # Apply transformations
                                    transformed_points = transform_points_combined(
                                        target_points, angle, scale_x, scale_y, skew_x, skew_y, stretch_x, stretch_y, translate_x, translate_y, centroid
                                    )
                                    
                                    # Calculate distance
                                    distance = calculate_distance(reference_points, transformed_points)
                                    
                                    # Update best parameters if this combination is better
                                    if distance < best_distance:
                                        best_distance = distance
                                        best_params = (angle, scale_x, scale_y, skew_x, skew_y, stretch_x, stretch_y, translate_x, translate_y)

# Print the best parameters and distance
print("Best parameters:", best_params)
print("Best distance:", best_distance)



def get_affine_matrix_combined(angle, scale_x, scale_y, skew_x, skew_y, stretch_x, stretch_y, translate_x, translate_y, centroid):
    """
    Compute the 2x3 affine transformation matrix for warpAffine, combining all transformations.
    """
    # Translate to origin
    translation_to_origin = np.array([
        [1, 0, -centroid[0]],
        [0, 1, -centroid[1]],
        [0, 0, 1]
    ])
    
    # Rotation matrix
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    
    # Scaling matrix
    scaling_matrix = np.array([
        [scale_x, 0, 0],
        [0, scale_y, 0],
        [0, 0, 1]
    ])
    
    # Skew matrix
    skew_matrix = np.array([
        [1, skew_x, 0],
        [skew_y, 1, 0],
        [0, 0, 1]
    ])
    
    # Stretch matrix
    stretch_matrix = np.array([
        [stretch_x, 0, 0],
        [0, stretch_y, 0],
        [0, 0, 1]
    ])
    
    # Translate back from origin
    translation_from_origin = np.array([
        [1, 0, centroid[0] + translate_x],
        [0, 1, centroid[1] + translate_y],
        [0, 0, 1]
    ])
    
    # Combine transformations
    transformation_matrix = translation_from_origin @ rotation_matrix @ scaling_matrix @ skew_matrix @ stretch_matrix @ translation_to_origin
    
    # Extract the 2x3 part of the matrix for warpAffine
    affine_matrix = transformation_matrix[:2, :]
    return affine_matrix

# Compute the affine matrix
affine_matrix = get_affine_matrix_combined(*best_params, centroid)

# Apply the transformation to the image
transformed_image = cv2.warpAffine(target_image, affine_matrix, (target_image.shape[1], target_image.shape[0]))

# Plot reference image
plt.subplot(2, 3, 1)
plt.imshow(reference_image)
plt.scatter(np.array(reference_points)[:, 0], np.array(reference_points)[:, 1], c='blue', label='Reference Points')
plt.title('Reference Image')

# Plot transformed target image
plt.subplot(2, 3, 2)
plt.imshow(target_image)
plt.scatter(np.array(target_points)[:, 0], np.array(target_points)[:, 1], c='red', label='Target Points')
plt.title('Target Image')

plt.subplot(2, 3, 3)
plt.imshow(transformed_image)
plt.scatter(np.array(transformed_points)[:, 0], np.array(transformed_points)[:, 1], c='red', label='Transformed Points')
plt.title('Transformed Image')

plt.subplot(2, 3, 4)
plt.scatter(np.array(reference_points)[:, 0], np.array(reference_points)[:, 1], c='blue', label='Reference Points')
plt.scatter(np.array(target_points)[:, 0], np.array(target_points)[:, 1], c='red', label='Target Points')

plt.subplot(2, 3, 5)
plt.scatter(np.array(reference_points)[:, 0], np.array(reference_points)[:, 1], c='blue', label='Reference Points')
plt.scatter(np.array(transformed_points)[:, 0], np.array(transformed_points)[:, 1], c='red', label='Transformed Points')

plt.subplot(2, 3, 6)
plt.plot(distances, marker='o')
plt.xlabel('Iteration')
plt.ylabel('Distance')
plt.title('Optimization Progress')
plt.grid(True)
plt.show()