from typing import Optional, Tuple, Dict

import cv2
import numpy as np
from numpy.typing import NDArray

from debugging import ImageDebugger, NullImageDebugger


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
    # 6. chose 5 largest contours measured by length and enclosed area and simplify them using Douglas-Peucker
    contours = sorted(contours, key=lambda c: cv2.contourArea(c) + cv2.arcLength(c, True), reverse=True)[:5]
    simplified_contours = [cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True) for c in contours]
    debugImages.add_contour_image("Contours simplified", simplified_contours, edged.shape)
    # 7. Select largest simplified contour by the following policy:
    # compute how many pixels of the minimum enclosed rectangle are covered by the contour.
    # Filter out contours with a bad coverage as they are likely not similar to a rectangle.
    # From the remaining contours select the one with the largest enclosed area.
    coverage_threshold = 0.7
    candidates = []
    for i, c in enumerate(simplified_contours):
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

        if iou >= coverage_threshold:
            candidates.append({
                "area": cv2.contourArea(c),
                "box": box,
                "contour": c,
            })

    # Choose the rectangle with the largest area among those with good coverage
    if candidates:
        best_candidate = max(candidates, key=lambda x: x["area"])
        best_contour = best_candidate["contour"]
        best_box = best_candidate["box"]
        if len(best_contour) == 4:
            debugImages.add_contour_image("Best Bounds", [best_contour], edged.shape)
            return best_contour.reshape(4, 2) / scaling_factor
        else:
            debugImages.add_contour_image("Best Bounds", [best_box], edged.shape)
            return best_box.astype(np.float32) / scaling_factor

    return None


def crop_image(
    input_path: str,
    output_path: str,
    imageDebugger: ImageDebugger = NullImageDebugger()
) -> None:
    """
    Detects and crops the document from an input image and saves the result.

    Args:
        input_path: Path to input image.
        output_path: Path to save cropped image.
        imageDebugger: Object for debugging intermediate steps.
    """
    # Reading the input image
    image = cv2.imread(input_path)
    if image is None:
        print(f"Could not read image {input_path}")
        imageDebugger.plot()
        return

    imageDebugger.add_image("Original", image)

    # Find the contour of the document
    doc_cnt = find_document_contour(image, imageDebugger)
    if doc_cnt is None:
        print(f"Could not find document contour in {input_path}.")
        imageDebugger.plot()
        return

    # Warp the image to contain only the part inside the 4 points
    warped = four_point_transform(image, doc_cnt)
    cv2.imwrite(output_path, warped)
    print(f"Cropped image saved to {output_path}")
    imageDebugger.plot()
