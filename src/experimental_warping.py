from typing import List

import cv2
import numpy as np
from numpy.typing import NDArray

from .debugging import ImageDebugger, NullImageDebugger


def get_corners(contour: NDArray[np.float32]) -> NDArray[np.float32]:
    """
    Approximates the contour to 4 corners and returns them ordered as:
    top-left, bottom-left, bottom-right, top-right
    """
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)

    if len(approx) != 4:
        raise ValueError("Contour does not have exactly 4 corners.")

    s = approx.sum(axis=1)
    diff = np.diff(approx, axis=1)
    return np.array([
        approx[np.argmin(s)],     # top-left
        approx[np.argmax(diff)],  # bottom-left
        approx[np.argmax(s)],     # bottom-right
        approx[np.argmin(diff)]   # top-right
    ], dtype=np.float32)


def extract_side(
    contour: NDArray[np.float32],
    start: NDArray[np.float32],
    end: NDArray[np.float32]
) -> NDArray[np.float32]:
    """
    Extracts the section of the contour path between two corner points.
    """
    start_idx = np.argmin(np.linalg.norm(contour - start, axis=1))
    end_idx = np.argmin(np.linalg.norm(contour - end, axis=1))

    if start_idx < end_idx:
        side = contour[start_idx:end_idx + 1]
    else:
        side = np.concatenate([contour[start_idx:], contour[:end_idx + 1]])

    return side


def cumulative_lengths(points: NDArray[np.float32]) -> NDArray[np.float32]:
    """
    Computes the cumulative distance along a polyline.
    """
    dists = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate([[0], np.cumsum(dists)])


def get_dst_points(
    start: NDArray[np.float32],
    end: NDArray[np.float32],
    relative_segment_lengths: NDArray[np.float32],
) -> NDArray[np.float32]:
    return np.array([(1-length) * start + length * end for length in relative_segment_lengths])


def warp_contour_to_rect(
    image: NDArray[np.uint8],
    contour: NDArray[np.float32],
    imageDebugger: ImageDebugger = NullImageDebugger()
) -> NDArray[np.uint8]:
    """
    Warps a distorted rectangular contour into a flat, axis-aligned rectangle
    using Thin Plate Spline (TPS) transformation.

    Args:
        image: Input image.
        contour: (N, 2) array representing the contour points.
        imageDebugger: a class to debug intermediate images

    Returns:
        Warped image of the rectified region.
    """
    # Step 1: Approximate and order corners
    corners = get_corners(contour)  # tl, bl, br, tr
    tl, bl, br, tr = corners
    imageDebugger.add_points("all_points", np.int64(contour), image.shape)
    imageDebugger.add_points("corners", np.int64(corners), image.shape)

    # Step 2: Define target rectangle corners
    width = int(max(np.linalg.norm(tl - tr), np.linalg.norm(bl - br)))
    height = int(max(np.linalg.norm(tl - bl), np.linalg.norm(tr - br)))

    dst_corners = np.array([
        [0, 0],
        [0, height - 1],
        [width - 1, height - 1],
        [width - 1, 0]
    ], dtype=np.float32)

    # Step 3: Sample along each contour side and map to straight edges
    src_points: List[NDArray[np.float32]] = []
    dst_points: List[NDArray[np.float32]] = []

    for i in range(4):
        start_corner = corners[i]
        end_corner = corners[(i + 1) % 4]

        src_side = extract_side(contour, start_corner, end_corner)
        segment_lengths = cumulative_lengths(src_side)
        segment_lengths /= segment_lengths[-1]
        dst_side = get_dst_points(dst_corners[i], dst_corners[(i + 1) % 4], segment_lengths)
        assert len(dst_side) == len(src_side)
        imageDebugger.add_points(f"Side {i}", np.int64(src_side), image.shape)
        imageDebugger.add_points(f"Side {i} corrected", np.int64(dst_side), image.shape)

        src_points.append(src_side[:-1])
        dst_points.append(dst_side[:-1])

    src_pts_all = np.concatenate(src_points).reshape(1, -1, 2)
    dst_pts_all = np.concatenate(dst_points).reshape(1, -1, 2)

    # Step 4: Create match list (point i in src_pts_all corresponds to point i in dst_pts_all)
    matches = [cv2.DMatch(i, i, 0) for i in range(src_pts_all.shape[1])]

    # Step 5: Thin Plate Spline transform
    tps = cv2.createThinPlateSplineShapeTransformer()
    tps.estimateTransformation(dst_pts_all, src_pts_all, matches)
    warped = tps.warpImage(image)

    return warped[:height, :width, :]