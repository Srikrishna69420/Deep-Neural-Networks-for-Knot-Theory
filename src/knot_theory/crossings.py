"""Image preprocessing and skeleton-graph crossing candidate detection."""

from pathlib import Path
import cv2
import networkx as nx
import numpy as np
from skimage.morphology import skeletonize


def preprocess_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not load image: {path}")
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def skeletonize_image(binary):
    return skeletonize(binary > 0).astype(np.uint8)


def skeleton_graph(skeleton):
    graph = nx.Graph()
    height, width = skeleton.shape
    foreground_y, foreground_x = np.nonzero(skeleton)
    graph.add_nodes_from((int(x), int(y)) for y, x in zip(foreground_y, foreground_x))
    neighbours = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for y, x in zip(foreground_y, foreground_x):
        for dx, dy in neighbours:
            nx_, ny_ = x + dx, y + dy
            if 0 <= nx_ < width and 0 <= ny_ < height and skeleton[ny_, nx_]:
                graph.add_edge((int(x), int(y)), (int(nx_), int(ny_)))
    return graph


def find_crossings(graph, radius=5):
    if radius < 0:
        raise ValueError("radius must be non-negative")
    candidates = [node for node, degree in graph.degree() if degree >= 4]
    filtered = []
    for candidate in candidates:
        if all(np.hypot(candidate[0] - kept[0], candidate[1] - kept[1]) > radius for kept in filtered):
            filtered.append(candidate)
    return filtered


def estimate_crossing_number(image_path, radius=5):
    binary = preprocess_image(image_path)
    skeleton = skeletonize_image(binary)
    graph = skeleton_graph(skeleton)
    return len(find_crossings(graph, radius)), graph, skeleton
