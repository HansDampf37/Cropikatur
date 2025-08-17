from typing import Optional

import cv2
import numpy as np
from numpy.typing import NDArray

from .debugging import ImageDebugger


def find_document_contour(
    image: NDArray[np.uint8],
    debugImages: ImageDebugger
) -> Optional[NDArray[np.float32]]:
    """
    Detects the document-like contour in the image.

    Args:
        image: Input BGR image.
        debugImages: Debug object to store intermediate images.

    Returns:
        Array of 4 corner points representing the detected document, or None if not found.
    """
    # 1. convert to Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    debugImages.add_image("Gray", gray)
    # 2. scale image down to a fixed size
    img_width, img_height = gray.shape
    scaling_factor = 300 / max(img_width, img_height)
    scaled_down = cv2.resize(gray, (0, 0), fx=scaling_factor, fy=scaling_factor)
    debugImages.add_image("ScaledDown", scaled_down)
    # 3. blur with gaussian kernel
    blurred = cv2.GaussianBlur(scaled_down, (7, 7), 0)
    debugImages.add_image("Blurred", blurred)
    # 4. apply canny edge detection
    edged = cv2.Canny(blurred, 30, 100)
    debugImages.add_image("Edges", edged)
    # 5. find contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    debugImages.add_contour_image("Contours", contours, edged.shape)
    # 6. chose 5 largest contours measured by length and enclosed area
    contours = sorted(contours, key=lambda c: cv2.contourArea(c) + cv2.arcLength(c, True), reverse=True)[:5]
    # 7. Select largest simplified contour by the following policy:
    # compute how many pixels of the minimum enclosed rectangle are covered by the contour.
    # Filter out contours with a bad coverage as they are likely not similar to a rectangle.
    # From the remaining contours select the one with the largest enclosed area.
    coverage_threshold = 0.9
    best_contour = None
    best_contour_area = 0
    for i, c in enumerate(contours):
        debugImages.add_contour_image(f"{i}: {len(c)} corners", [c], edged.shape)
        best_enclosing_rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(best_enclosing_rect)
        box = np.int32(box)

        # Compute coverage between rectangle and contour
        rect_mask = np.zeros(edged.shape, dtype=np.uint8)
        contour_mask = np.zeros(edged.shape, dtype=np.uint8)
        cv2.drawContours(rect_mask, [box], -1, 255, thickness=-1)
        cv2.drawContours(contour_mask, [c], -1, 255, thickness=-1)

        intersection = np.logical_and(rect_mask, contour_mask).sum()
        union = np.logical_or(rect_mask, contour_mask).sum()
        iou = intersection / union if union > 0 else 0

        if iou >= coverage_threshold and cv2.contourArea(c) > best_contour_area:
            best_contour = c
            best_contour_area = cv2.contourArea(c)

    if best_contour is not None:
        debugImages.add_contour_image("Best Bounds", [best_contour], edged.shape)
        return np.int32(best_contour / scaling_factor)

    return None