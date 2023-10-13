import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
img = cv2.imread("multiplicity_regions_ais_cstr2.JPG")

# Convert the image to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply adaptive thresholding
adaptive_binary = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY_INV, 11, 2)

# Apply Otsu's thresholding
_, otsu_binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Combine the two binary images using logical OR
combined_binary = cv2.bitwise_or(adaptive_binary, otsu_binary)

# Use Canny edge detection
edges = cv2.Canny(combined_binary, 50, 150)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter out small contours and the x-axis
filtered_contours = []
for contour in contours:
    if cv2.contourArea(contour) > 100:  # Filtering based on contour area
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w)/h
        if not (aspect_ratio > 5 and y > gray_img.shape[0] - 50):  # Filtering the x-axis
            filtered_contours.append(contour)

# Convert contours to data coordinates
def pixel_to_data(pixel_coord, axis_limits, img_shape):
    x_data = axis_limits[0][0] + pixel_coord[0] * (axis_limits[0][1] - axis_limits[0][0]) / img_shape[1]
    y_data = axis_limits[1][1] - pixel_coord[1] * (axis_limits[1][1] - axis_limits[1][0]) / img_shape[0]
    return x_data, y_data

axis_limits = [[-0.1, 1.5], [0.22, 1.25]]
data_contours = []
for contour in filtered_contours:
    data_contour = [pixel_to_data(pt[0], axis_limits, gray_img.shape) for pt in contour]
    data_contours.append(data_contour)

# Visualizing the contours on the original image for verification (optional)
for contour in filtered_contours:
    cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

u_values = np.array([
    [0, 0.41200145787774667],
    [0, 0.4237849779086895],
    [0, 0.4399837848291458],
    [0, 0.45618556701030943],
    [0, 0.475329882],
    [0, 0.49300441826215036],
    [0, 0.5092032251826066],
    [0, 0.5254050073637704],
    [0, 0.5430780559646541],
    [0, 0.5607496169351841],
    [0, 0.5769499114859941],
    [0, 0.5946229600868776],
    [0.009090909, 0.61525193]])

aux = np.array(data_contour)
final_data =  np.vstack([u_values, aux])
plt.figure()
plt.scatter(final_data[:,0], final_data[:,1])

np.save(final_data)