import cv2
import numpy as np
import os

class Bounce_Detection:
    def __init__(self, table_corners):
        """
        Initialize Ball_Detection with table corners.
        :param table_corners: A list of four tuples representing the corners of the table [(x1, y1), (x2, y2), (x3, y3), (x4, y4)].
        """
        self.table_corners = np.array(table_corners, dtype=np.int32)
        self.table_mask =  np.zeros((1080, 1920), dtype=np.uint8)  # Assuming a 1920x1080 image
        cv2.fillPoly(self.table_mask, [self.table_corners], 1)

    def point_in_table(self, point):
        """
        Check if a point is inside the table area defined by the corners.

        :param point: A tuple (x, y) representing the ball's position.
        :return: True if the point is inside the table, False otherwise.
        """
        # Create a mask for the table polygon
        # Check if the point is inside the mask
        return self.table_mask[point[1], point[0]] == 1

    def bounce_detection(self, ball_coordinates):
        """
        Detect bounces based on ball coordinates within the table area.

        :param ball_coordinates: List of tuples [(x1, y1), (x2, y2), ...] representing ball positions over time.
        :return: A list of indices where the ball bounces within the table area.
        """
        bounces = []

        for i in range(1, len(ball_coordinates) - 1):
            prev_y = ball_coordinates[i - 1][1]
            curr_y = ball_coordinates[i][1]
            next_y = ball_coordinates[i + 1][1]
            y_change_prev = abs(curr_y - prev_y)
            y_change_next = abs(next_y - curr_y)

            # Check if the current point is within the table area
            if self.point_in_table(ball_coordinates[i]):
                # Case 1: Bounce on one side of the table
                if prev_y < curr_y >= next_y and y_change_prev < 25 and y_change_next < 25:
                    bounces.append(i)
                # Case 2: Bounce on the other side of the table
                elif prev_y > curr_y <= next_y and y_change_prev < 25 and y_change_next < 25:
                    bounces.append(i)
                elif prev_y == curr_y == next_y:
                    bounces.append(i)

        return bounces

# Example usage
if __name__ == "__main__":
    table_corners = [(734, 397), (1119, 399), (1150, 581), (742, 577)]
    ball_detector = Bounce_Detection(table_corners)

    ball_positions = [(800, 350), (900, 450), (950, 500), (1000, 450), (1100, 400)]  # Example coordinates
    bounces = ball_detector.bounce_detection(ball_positions)
    print("Bounce indices:", bounces)
