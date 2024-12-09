import cv2
import numpy as np

def map_ball_to_table(image_corners, table_corners, ball_position):
    """
    Map a ball's position from image frame to table coordinates.
    
    Args:
        image_corners (list of tuples): Four corners of the table in the image frame.
        table_corners (list of tuples): Four corners of the table in the table coordinate system.
        ball_position (tuple): (x, y) position of the ball in the image frame.

    Returns:
        tuple: (x, y) position of the ball in table coordinates.
    """
    # Convert corners to numpy arrays
    image_corners_np = np.array(image_corners, dtype="float32")
    table_corners_np = np.array(table_corners, dtype="float32")

    # Compute perspective transform matrix
    M = cv2.getPerspectiveTransform(image_corners_np, table_corners_np)

    # Transform the ball position
    ball_position_np = np.array([[ball_position]], dtype="float32")  # Shape (1, 1, 2)
    transformed_position = cv2.perspectiveTransform(ball_position_np, M)

    return tuple(transformed_position[0][0])  # Convert back to tuple


if __name__ == '__main__':
    # Image corners (selected by user)
    image_corners = [(736, 397), (200, 100), (500, 400), (100, 400)]

    # Table coordinates (top-down view)
    table_corners = [(0, 0), (1000, 0), (1000, 500), (0, 500)]

    # Ball position in the image
    ball_position = (300, 250)

    # Map the ball to table coordinates
    table_position = map_ball_to_table(image_corners, table_corners, ball_position)
    print(f"Ball position in table coordinates: {table_position}")