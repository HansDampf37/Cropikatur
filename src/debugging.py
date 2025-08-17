from typing import Dict, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt


class ImageDebugger:
    """
    Utility class to store and visualize intermediate images during processing.
    Used for debugging and understanding internal stages of the pipeline.
    """

    def __init__(self):
        self.intermediate_images: Dict[str, NDArray[np.uint8]] = {}

    def add_image(self, name: str, image: NDArray[np.uint8]) -> None:
        """Stores a single intermediate image under a name."""
        self.intermediate_images[name] = image

    def add_contour_image(self, name: str, contours: NDArray[np.int32], shape: Tuple[int, int]) -> None:
        """
        Stores a visual representation of given contours drawn on a black image.

        Args:
            name: Label for the image.
            contours: Contour array (as returned by cv2.findContours or similar).
            shape: Shape of the blank canvas (usually the source image shape).
        """
        contour_image = cv2.drawContours(np.zeros(shape, dtype=np.uint8), contours, -1, (255,), -1)
        self.add_image(name, contour_image)

    def add_points(self, name: str, points: NDArray, shape: Tuple[int]):
        points_image = np.zeros(shape=shape)
        # Draw points as circles
        for pt in points:
            cv2.circle(points_image, tuple(pt), radius=11, color=(1.0,), thickness=-1)
        self.add_image(name, points_image)


    def plot(self) -> None:
        """Plots all stored intermediate images using matplotlib."""
        for name, image in self.intermediate_images.items():
            plt.imshow(image, cmap="gray")
            plt.axis('off')
            plt.title(name)
            plt.show()


class NullImageDebugger(ImageDebugger):
    """
    Null object for ImageDebugger.
    Methods do nothing, allowing easy disabling of debugging without modifying logic.
    """

    def add_image(self, name: str, image: NDArray[np.uint8]) -> None:
        pass

    def add_contour_image(self, name: str, contours: NDArray[np.int32], shape: Tuple[int, int]) -> None:
        pass

    def plot(self) -> None:
        pass