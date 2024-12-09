import cv2
import sys
import os
import numpy as np
sys.path.append('../')

import matplotlib.pyplot as plt

def read_img(img_path):
    image = cv2.imread(img_path)  # BGR

    return image

def draw_edge_contours(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection using Canny
    edges = cv2.Canny(blurred, 50, 150)

    # Hough Transform
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)

    # Draw lines on the image
    image_with_lines = image.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image_with_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    height, width = gray.shape
    table_contours = contour_filtering(contours, height, width)
    # table_contours = contour_filtering_with_lines(edges, contours, height, width)
    if table_contours is None:
        print("Table contour is None")
        table_contours = contours

    # Draw contours on the original image
    image_with_contours = image.copy()
    cv2.drawContours(image_with_contours, table_contours, -1, (0, 255, 0), 2)

    return edges, image_with_contours, image_with_lines



def contour_filtering(contours, height, width):
    image_center = (width//2, height//2)
    table_contours = []
    largest_area = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        if 4.0< aspect_ratio < 5.0 :
            M = cv2.moments(contour)
            if M["m00"] > 0:  # Avoid division by zero
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Check if the centroid is near the image center
                distance = ((cx - image_center[0]) ** 2 + (cy - image_center[1]) ** 2) ** 0.5
                max_distance = min(width, height) // 4  # Adjust threshold as needed
                if distance < max_distance:
                    table_contours.append(contour)
    print(len(table_contours))
    return table_contours


def contour_filtering_with_lines(edges, contours, height, width):
   

    # Hough Transform to detect straight lines
    lines = cv2.HoughLines(edges, rho=1, theta=np.pi / 180, threshold=100)

    # Filter contours based on straight-line alignment
    image_center = (width // 2, height // 2)  # Center of the image
    table_contours = []

    for contour in contours:
        # Calculate bounding box and aspect ratio
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h

        # Check if contour is near the center of the image
        M = cv2.moments(contour)
        if M["m00"] > 0:  # Avoid division by zero
            cx = int(M["m10"] / M["m00"])  # Contour's centroid X
            cy = int(M["m01"] / M["m00"])  # Contour's centroid Y
            distance = ((cx - image_center[0]) ** 2 + (cy - image_center[1]) ** 2) ** 0.5

            # Maximum allowable distance from the center
            max_center_distance = min(width, height) // 4  # Adjust as needed
            if distance > max_center_distance:
                continue  # Skip contours far from the center

        if aspect_ratio > 2.0:  # Filter based on aspect ratio
            # Check if the contour's points align with straight lines
            matches = 0
            print("here")
            if lines is not None:  # Ensure lines are detected
                for line in lines:
                    rho, theta = line[0]
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * a)
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * a)

                    # Check if contour points align with the line
                    for point in contour:
                        px, py = point[0]
                        distance_to_line = abs(a * px + b * py - rho) / (a**2 + b**2) ** 0.5
                        if distance_to_line < 5:  # Threshold for point-to-line distance
                            matches += 1
                            break

            # If a significant portion of the contour aligns with straight lines, keep it
            if matches > len(contour) * 0.3:  # At least 30% of points match
                table_contours.append(contour)

    print(f"Filtered {len(table_contours)} contours near the center with straight-line alignment.")
    return table_contours



def line_filtering(lines):
    # Separate horizontal and vertical lines
    horizontal_lines = []
    vertical_lines = []

    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            # Check if the line is horizontal or vertical
            if abs(theta) < np.pi / 6 or abs(theta - np.pi) < np.pi / 6:  # Horizontal
                horizontal_lines.append((rho, theta))
            elif abs(theta - np.pi / 2) < np.pi / 6:  # Vertical
                vertical_lines.append((rho, theta))
    
    return horizontal_lines, vertical_lines


def detect_table_with_edges(image):
    # Preprocess the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Detect lines using Hough Transform
    lines = cv2.HoughLines(edges, rho=1, theta=np.pi / 180, threshold=100)

    # Separate horizontal and vertical lines
    horizontal_lines = []
    vertical_lines = []
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            if abs(theta) < np.pi / 6 or abs(theta - np.pi) < np.pi / 6:  # Horizontal
                horizontal_lines.append((rho, theta))
            elif abs(theta - np.pi / 2) < np.pi / 6:  # Vertical
                vertical_lines.append((rho, theta))

    # Find intersections of horizontal and vertical lines
    def line_intersection(line1, line2):
        rho1, theta1 = line1
        rho2, theta2 = line2

        A = np.array([[np.cos(theta1), np.sin(theta1)],
                      [np.cos(theta2), np.sin(theta2)]])
        b = np.array([[rho1], [rho2]])

        # Solve linear equations to find the intersection
        x, y = np.linalg.solve(A, b)
        return int(x), int(y)

    # Compute corners from intersections
    corners = []
    for h_line in horizontal_lines:
        for v_line in vertical_lines:
            intersection = line_intersection(h_line, v_line)
            corners.append(intersection)

    # Filter corners that are too close or too far from the center
    height, width = gray.shape
    image_center = (width // 2, height // 2)
    max_distance = min(width, height) // 2
    valid_corners = []
    for corner in corners:
        cx, cy = corner
        distance = ((cx - image_center[0]) ** 2 + (cy - image_center[1]) ** 2) ** 0.5
        if distance < max_distance:
            valid_corners.append(corner)

    # Draw the detected corners and lines
    image_with_edges = image.copy()
    for corner in valid_corners:
        cv2.circle(image_with_edges, corner, 10, (0, 0, 255), -1)  # Draw corners in red

    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * a)
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * a)
            cv2.line(image_with_edges, (x1, y1), (x2, y2), (0, 255, 0), 1)  # Green lines

    return image_with_edges, valid_corners


def get_user_selected_corners_with_matplotlib(image):
    """Allow the user to select corners using Matplotlib."""
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Select four corners (close the window after selecting)")
    points = plt.ginput(4)  # User selects 4 points
    plt.close()

    # Convert points to integer tuples
    return [(int(x), int(y)) for x, y in points]

if __name__ == '__main__':
    output_folder = "/home/s224705071/github/PhysicsInformedDeformableAttentionNetwork/results/demo/logs/output"
    image = read_img("/home/s224705071/github/PhysicsInformedDeformableAttentionNetwork/data/tta_dataset/images/img_000000.jpg")
    print(f"Image shape: {image.shape if image is not None else 'Image not loaded'}")

    edges, image_with_contours, image_with_lines = draw_edge_contours(image)

    selected_corners = get_user_selected_corners_with_matplotlib(image)
    if selected_corners:
        print("Selected corners:", selected_corners)
        # Draw the selected corners and connect them
        for point in selected_corners:
            cv2.circle(image, point, 10, (0, 255, 0), -1)
        cv2.polylines(image, [np.array(selected_corners, dtype=np.int32)], isClosed=True, color=(255, 0, 0), thickness=2)

        # Save the result
        cv2.imwrite("/path/to/output_with_corners.jpg", image)
        print("Result saved with selected corners.")


    # Save the images to the specified folder
    original_image_path = os.path.join(output_folder, "original_image.jpg")
    edges_image_path = os.path.join(output_folder, "edges_detected.jpg")
    contours_image_path = os.path.join(output_folder, "contours_image.jpg")
    lines_image_path = os.path.join(output_folder, "line_image.jpg")

    # Save images
    cv2.imwrite(original_image_path, image)
    cv2.imwrite(edges_image_path, edges)
    cv2.imwrite(contours_image_path, image_with_contours)
    cv2.imwrite(lines_image_path, image_with_lines)

    print(f"Images saved to {output_folder}")