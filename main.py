import cv2
import shutil
import json
import numpy as np
from collections import Counter
from pathlib import Path

UNKNOWN_THRESHOLD = 0.5

def debug_arrow_tip(contour):

    img = np.full(
        (600, 600, 3),
        255,
        dtype=np.uint8
    )

    approx = cv2.approxPolyDP(
        contour,
        0.02 * cv2.arcLength(contour, True),
        True
    )

    points = approx.reshape(-1, 2)

    smallest_angle = 360
    tip = None

    for i in range(len(points)):

        p_prev = points[i - 1]
        p = points[i]
        p_next = points[(i + 1) % len(points)]

        v1 = p_prev - p
        v2 = p_next - p

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            continue

        angle = np.degrees(
            np.arccos(
                np.clip(
                    np.dot(v1, v2) /
                    (norm1 * norm2),
                    -1,
                    1
                )
            )
        )

        if angle < smallest_angle:

            smallest_angle = angle
            tip = p

    cv2.drawContours(
        img,
        [contour],
        -1,
        (0, 0, 0),
        2
    )

    print("Tip:", tip)
    print("Smallest angle:", smallest_angle)
    print("Points:")
    print(points)

    if tip is not None:

        cv2.circle(
            img,
            tuple(tip),
            10,
            (0, 0, 255),
            -1
        )

    cv2.imshow(
        "Arrow Tip",
        img
    )

    cv2.waitKey(0)

def save_symbol_example(
    image_path,
    symbol_name,
    folder="symbols"
):
    symbol_dir = Path(folder) / symbol_name

    symbol_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    metadata_file = (
        symbol_dir /
        "metadata.json"
    )

    if not metadata_file.exists():

        metadata = {
            "name": symbol_name,
            "components": {},
            "examples": 0
        }

        with open(
            metadata_file,
            "w"
        ) as f:

            json.dump(
                metadata,
                f,
                indent=4
            )

    existing = list(
        symbol_dir.glob(
            f"{symbol_name}*.png"
        )
    )

    next_num = len(existing) + 1

    destination = (
        symbol_dir /
        f"{symbol_name}{next_num}.png"
    )

    shutil.copy(
        image_path,
        destination
    )

    print(
        f"Saved: {destination}"
    )
def update_symbol_metadata(
    symbol_name,
    components,
    folder="symbols"
):
    metadata_file = (
        Path(folder) /
        symbol_name /
        "metadata.json"
    )

    counts = Counter(
        components
    )

    metadata = {
        "name": symbol_name,
        "components": dict(counts),
        "examples": len(
            list(
                (
                    Path(folder) /
                    symbol_name
                ).glob("*.png")
            )
        )
    }

    with open(
        metadata_file,
        "w"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4
        )

    print(
        f"Updated metadata for "
        f"{symbol_name}"
    )
def load_symbol_metadata(
    folder="symbols"
):
    metadata = {}

    root = Path(folder)

    if not root.exists():
        return metadata

    for symbol_dir in root.iterdir():

        if not symbol_dir.is_dir():
            continue

        file = (
            symbol_dir /
            "metadata.json"
        )

        if file.exists():

            with open(file) as f:

                metadata[
                    symbol_dir.name
                ] = json.load(f)

    return metadata
def classify_compound_symbol(
    component_names,
    metadata_db
):
    detected = Counter(
        component_names
    )

    best_name = None
    best_score = float("inf")

    for name, meta in metadata_db.items():

        target = Counter(
            meta["components"]
        )

        score = 0

        all_keys = set(
            detected.keys()
        ) | set(
            target.keys()
        )

        for key in all_keys:

            score += abs(
                detected.get(key, 0)
                -
                target.get(key, 0)
            )

        if score < best_score:

            best_score = score
            best_name = name

    return best_name, best_score
def learn_if_unknown(
    image_path,
    best_name,
    best_score,
    folder="symbols"
):
    if best_score < UNKNOWN_THRESHOLD:
        return

    print(
        f"Unknown symbol detected "
        f"(closest: {best_name}, score={best_score:.3f})"
    )

    new_name = input(
        "Enter symbol name: "
    ).strip()

    if not new_name:
        print("Cancelled.")
        return

    save_symbol_example(
        image_path,
        new_name,
        folder
    )

def get_arrow_rotation(contour):

    approx = cv2.approxPolyDP(
        contour,
        0.02 * cv2.arcLength(contour, True),
        True
    )

    points = approx.reshape(-1, 2)

    if len(points) < 3:
        return None

    smallest_angle = 360
    tip = None

    for i in range(len(points)):

        p_prev = points[i - 1]
        p = points[i]
        p_next = points[(i + 1) % len(points)]

        v1 = p_prev - p
        v2 = p_next - p

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            continue

        cos_angle = np.dot(v1, v2) / (
            norm1 * norm2
        )

        cos_angle = np.clip(
            cos_angle,
            -1,
            1
        )

        angle = np.degrees(
            np.arccos(cos_angle)
        )

        if angle < smallest_angle:

            smallest_angle = angle
            tip = p

    if tip is None:
        return None

    moments = cv2.moments(contour)

    if moments["m00"] == 0:
        return None

    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]

    dx = tip[0] - cx
    dy = tip[1] - cy

    rotation = np.degrees(
        np.arctan2(dy, dx)
    )

    return (rotation + 180) % 360

def classify_components(contours, library):
    results = []

    for contour in contours:

        if cv2.contourArea(contour) < 50:
            continue

        name, score = classify(
            contour,
            library
        )

        x, y, w, h = cv2.boundingRect(
            contour
        )

        results.append({
            "name": name,
            "score": score,
            "bbox": (x, y, w, h)
        })

    return results

def get_contour(image_path):
    img = cv2.imread(str(image_path))

    if img is None:
        raise FileNotFoundError(image_path)

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    _, thresh = cv2.threshold(
        gray,
        127,
        255,
        cv2.THRESH_BINARY_INV
    )

    contours, hierarchy = cv2.findContours(
        thresh,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contours = [
        c for c in contours
        if cv2.contourArea(c) > 50
    ]

    return contours


def load_symbol_library(folder="symbols"):

    library = {}

    folder_path = Path(folder)

    if not folder_path.exists():
        return library

    for symbol_dir in folder_path.iterdir():

        if not symbol_dir.is_dir():
            continue

        name = symbol_dir.name

        metadata_file = (
            symbol_dir /
            "metadata.json"
        )

        if not metadata_file.exists():

            with open(
                metadata_file,
                "w"
            ) as f:

                json.dump(
                    {
                        "name": name,
                        "components": {},
                        "examples": 0
                    },
                    f,
                    indent=4
                )
            
        library[name] = []

        for file in symbol_dir.glob("*.png"):

            try:

                contours = get_contour(
                    file
                )

                if not contours:
                    continue

                largest = max(
                    contours,
                    key=cv2.contourArea
                )

                library[name].append(
                    largest
                )

            except Exception as e:

                print(
                    f"Failed to load {file}: {e}"
                )

    return library


def classify(contour, library):
    scores = []

    for name, templates in library.items():

        for template in templates:

            score = cv2.matchShapes(
                contour,
                template,
                cv2.CONTOURS_MATCH_I1,
                0
            )

            scores.append(
                (name, score)
            )

    scores.sort(
        key=lambda x: x[1]
    )

    best_name, best_score = scores[0]

    return best_name, best_score


def get_top_matches(
    contour,
    library,
    limit=5
):
    scores = []

    for name, templates in library.items():

        best_score = float("inf")

        for template in templates:

            score = cv2.matchShapes(
                contour,
                template,
                cv2.CONTOURS_MATCH_I1,
                0
            )

            best_score = min(
                best_score,
                score
            )

        scores.append(
            (name, best_score)
        )

    scores.sort(
        key=lambda x: x[1]
    )

    return scores[:limit]


def classify_with_rotation(
    contour,
    library
):
    name, score = classify(
        contour,
        library
    )

    return name, score, None


def corner_count(
    contour,
    approx=0.04
):
    epsilon = (
        approx *
        cv2.arcLength(
            contour,
            True
        )
    )

    simplified = cv2.approxPolyDP(
        contour,
        epsilon,
        True
    )

    return len(
        simplified
    )

def rotation_to_direction(angle):

    directions = [
        "→",
        "↘",
        "↓",
        "↙",
        "←",
        "↖",
        "↑",
        "↗"
    ]

    index = round(
        angle / 45
    ) % 8

    return directions[index]

if __name__ == "__main__":

    library = load_symbol_library()

    if not library:

        print(
            "No symbols loaded. "
            "Create folders in symbols/ first."
        )

        exit()

    contours = get_contour(
        "test.png"
    )

    components = classify_components(
        contours,
        library
    )

    print()
    print("Detected Components")
    print("-" * 40)

    for component in components:

        print(
            f"{component['name']:<20}"
            f"{component['score']:.6f}"
        )

    print("-" * 40)

    print()
    print("Detailed Component Data")
    print("-" * 40)

    for component in components:

        print(component)

    print("-" * 40)
    