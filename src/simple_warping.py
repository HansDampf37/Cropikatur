import cv2
import numpy as np
from numpy.typing import NDArray


def order_points(pts: NDArray[np.float32]) -> NDArray[np.float32]:
    """
    Orders 4 points in consistent top-left, top-right, bottom-right, bottom-left order.

    Args:
        pts: 4x2 array of points

    Returns:
        4x2 array of ordered points
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]      # top-left
    rect[2] = pts[np.argmax(s)]      # bottom-right
    rect[1] = pts[np.argmin(diff)]   # top-right
    rect[3] = pts[np.argmax(diff)]   # bottom-left

    return rect


def four_point_transform(image: NDArray[np.uint8], pts: NDArray[np.float32]) -> NDArray[np.uint8]:
    """
    Applies a perspective transform to extract the region bounded by pts.

    Args:
        image: Input image.
        pts: 4x2 array of corner points.

    Returns:
        Warped top-down view of the selected region.
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)

    maxWidth = int(max(widthA, widthB))
    maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))