from enum import Enum
from math import sqrt
from typing import Optional

import cv2
from numpy.typing import NDArray

from .debugging import ImageDebugger, NullImageDebugger
from .find_document_contour import find_document_contour
from .simple_warping import four_point_transform


class AspectRatio(Enum):
    A_x = "Ax"
    B_x = "Bx"
    C_x = "Cn"
    r16_9 = "16/9"
    r3_2 = "3/2"
    square = "square"

    def __str__(self):
        return self.value

    def __float__(self):
        if self == AspectRatio.A_x or self == AspectRatio.B_x:
            return sqrt(2)
        elif self == AspectRatio.C_x:
            return 229/114
        elif self == AspectRatio.r16_9:
            return 16/9
        elif self == AspectRatio.r3_2:
            return 3/2
        elif self == AspectRatio.square:
            return 1.0
        else:
            raise NotImplementedError("Aspect ratio not implemented")


def apply_aspect_ratio(image: NDArray, aspect_ratio: Optional[AspectRatio]) -> NDArray:
    if aspect_ratio is None:
        return image
    else:
        height = image.shape[0]
        width = image.shape[1]
        if width > height:
            if height * float(aspect_ratio) > width:
                return cv2.resize(image, (int(height * float(aspect_ratio)), height))
            else:
                return cv2.resize(image, (width, int(width / float(aspect_ratio))))
        else:
            if width * float(aspect_ratio) > height:
                return cv2.resize(image, (width, int(width * float(aspect_ratio))))
            else:
                return cv2.resize(image, (int(height / float(aspect_ratio)), height))


def crop_image(
    input_path: str,
    output_path: str,
    debug: bool = False,
    aspect_ratio: Optional[AspectRatio] = None
) -> None:
    """
    Detects and crops the document from an input image and saves the result.

    Args:
        input_path: Path to input image.
        output_path: Path to save cropped image.
        debug: Whether to debug intermediate results.
        aspect_ratio: Aspect ratio to use or None to keep the ratio that is produced after cropping and warping.
    """
    imageDebugger = ImageDebugger() if debug else NullImageDebugger()
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

    # Simplify contour slightly
    arc_len = cv2.arcLength(doc_cnt, True)
    simplified_contour = cv2.approxPolyDP(doc_cnt, 0.02 * arc_len, True)
    simplified_contour = simplified_contour.reshape(-1, 2)

    # Warp the image to contain only the part inside the contour
    warped = four_point_transform(image, simplified_contour)

    # Apply aspect ratio if specified
    final_image = apply_aspect_ratio(warped, aspect_ratio)

    # safe
    cv2.imwrite(output_path, final_image)
    print(f"Cropped image saved to {output_path}")
    imageDebugger.plot()
