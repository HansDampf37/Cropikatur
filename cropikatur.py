import argparse
import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray


class ImageDebugger:
    """Holds intermediate images produced during extraction of rectangle"""

    def __init__(self):
        self.intermediate_images = {}

    def add_image(self, name: str, image: NDArray[np.uint8]) -> None:
        """Adds intermediate image to this class"""
        self.intermediate_images[name] = image

    def add_contour_image(self, name: str, contours: NDArray[Tuple[int, int]], shape: Tuple[int, int]):
        """Adds contour image to this class"""
        contour_image = cv2.drawContours(np.zeros(shape), contours, -1, (255,), -1)
        self.add_image(name, contour_image)

    def plot(self):
        """Plots intermediate images"""
        import matplotlib.pyplot as plt
        for i, (name, image) in enumerate(self.intermediate_images.items()):
            plt.imshow(image)
            plt.axis('off')
            plt.title(name)
            plt.yticks(None)
            plt.xticks(None)
            plt.show()


class NullImageDebugger(ImageDebugger):
    """
    Null Object for IntermediateImages. Can be used to not log and plot any images during cropping process while also
    preventing checking if IntermediateImages is null.
    """

    def add_image(self, name: str, image: NDArray[np.uint8]) -> None:
        pass

    def add_contour_image(self, name: str, contours: NDArray[Tuple[int, int]], shape: Tuple[int, int]) -> None:
        pass

    def plot(self) -> None:
        pass


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute width and height of the new image
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))


def find_document_contour(image, debugImages: ImageDebugger) -> NDArray:
    # 1. convert to Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    debugImages.add_image("Gray", gray)
    # 2. scale image down to a fixed size
    img_width, img_height = gray.shape
    scaling_factor = 300 / (max(img_width, img_height))
    scaled_down = cv2.resize(gray, (0, 0), fx=scaling_factor, fy=scaling_factor)
    debugImages.add_image("ScaledDown", scaled_down)
    # 3. blur with gaussian kernel
    blurred = cv2.GaussianBlur(scaled_down, (7, 7), 0)
    debugImages.add_image("Blurred", blurred)
    # 4. apply canny edge detection
    edged = cv2.Canny(blurred, 50, 200)
    debugImages.add_image("Edges", edged)
    # 5. find contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    debugImages.add_contour_image("Contours", contours, edged.shape)
    # 6. chose 5 largest contours measured by length and enclosed area and simplify them using Douglas-Peucker
    contours = sorted(contours, key=lambda c: cv2.contourArea(c) + cv2.arcLength(c, True), reverse=True)[:5]
    simplified_contours = [cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True) for c in contours]
    debugImages.add_contour_image("Contours simplified", simplified_contours, edged.shape)
    # 7. Select largest simplified contour with 4 corners
    for c in simplified_contours:
        if len(c) == 4:
            debugImages.add_contour_image("Identified Contour", [c], edged.shape)
            return c.reshape(4, 2) / scaling_factor
    # 8. (Only as Fallback) If no 4-point contour found, consider only contours that fit the smallest enclosing rectangle nicely.
    # and select the largest one of these contours
    coverage_threshold = 0.7
    candidates = []
    for i, c in enumerate(simplified_contours):
        debugImages.add_contour_image(f"{i}: {len(c)} corners", [c], edged.shape)
        best_enclosing_rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(best_enclosing_rect)
        box = np.int32(box)

        width, height = best_enclosing_rect[1]
        area = width * height

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
                "area": area,
                "box": box,
            })

    # Choose the rectangle with the largest area among those with good coverage
    if candidates:
        best_candidate = max(candidates, key=lambda x: x["area"])  # choose by area
        best_box = best_candidate["box"]
        debugImages.add_contour_image("Best Rotated Rectangle", [best_box], edged.shape)
        return best_box.astype(np.float32) / scaling_factor

    return None


def crop_image(input_path: str, output_path: Optional[str] = None, imageDebugger: ImageDebugger = NullImageDebugger()):
    if output_path is None:
        splits = input_path.split(".")
        splits[-2] += "_cropped"
        output_path = ".".join(splits)

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


def main():
    parser = argparse.ArgumentParser(description="Crop image to detected paper edges.")
    parser.add_argument("input", help="Path to input image or folder")
    parser.add_argument("--debug", help="Show debug images", default=False)
    args = parser.parse_args()
    args.input = args.input if not args.input.endswith("/") else args.input[:-1]  # remove trailing /
    debugImages = ImageDebugger() if args.debug else NullImageDebugger()
    if args.input is not None:
        if Path(args.input).is_file():
            crop_image(args.input, imageDebugger=debugImages)
        else:
            # iterate over files in dir
            for filename in os.listdir(args.input):
                filepath = os.path.join(args.input, filename)
                if os.path.isfile(filepath):
                    if not filename.endswith(".jpg") and not filename.endswith(".png") and not filename.endswith(
                            ".jpeg"):
                        print(f"Skipping {filepath} due to format incompatibility.")
                        continue

                    output_dir = args.input + "_cropped"
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    output_file_path = os.path.join(output_dir, filename)
                    crop_image(filepath, output_file_path, imageDebugger=debugImages)


if __name__ == "__main__":
    main()
