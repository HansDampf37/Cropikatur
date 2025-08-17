import cv2

from .debugging import ImageDebugger, NullImageDebugger
from .find_document_contour import find_document_contour
from .simple_warping import four_point_transform


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

    # Simplify contour slightly
    arc_len = cv2.arcLength(doc_cnt, True)
    simplified_contour = cv2.approxPolyDP(doc_cnt, 0.02 * arc_len, True)
    simplified_contour = simplified_contour.reshape(-1, 2)

    # Warp the image to contain only the part inside the contour
    warped = four_point_transform(image, simplified_contour)
    cv2.imwrite(output_path, warped)
    print(f"Cropped image saved to {output_path}")
    imageDebugger.plot()
