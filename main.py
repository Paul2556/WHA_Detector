import cv2
import shutil
import json
import numpy as np
import tempfile
import tkinter as tk
from collections import Counter
from datetime import datetime
from pathlib import Path
UNKNOWN_THRESHOLD = 0.5
SHARED_STATUS_FILE = Path("classification_status.json")

def _create_tk_icon_image():
    icon_data = [
        "P3 16 16 255",
    ]
    for y in range(16):
        row = []
        for x in range(16):
            dx = x - 7.5
            dy = y - 7.5
            radius = 6.5
            if dx * dx + dy * dy <= radius * radius:
                row.append("0 120 215")
            else:
                row.append("230 230 230")
        icon_data.append(" ".join(row))
    icon_data = "\n".join(icon_data)
    return tk.PhotoImage(data=icon_data)

def _write_temp_icon_file():
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None
    icon_path = Path(tempfile.gettempdir()) / "wha_detector_icon.png"
    if icon_path.exists():
        return str(icon_path)
    img = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse((8, 8, 120, 120), fill=(0, 120, 215, 255))
    draw.ellipse((32, 32, 96, 96), fill=(255, 255, 255, 255))
    try:
        img.save(icon_path)
        return str(icon_path)
    except Exception:
        return None

def set_app_icon(root):
    try:
        image = _create_tk_icon_image()
        root.iconphoto(False, image)
        root._icon_image = image
    except Exception:
        pass
    icon_path = _write_temp_icon_file()
    if icon_path is None:
        return
    try:
        from AppKit import NSImage, NSApplication
        nsimage = NSImage.alloc().initWithContentsOfFile_(icon_path)
        if nsimage is not None:
            NSApplication.sharedApplication().setApplicationIconImage_(nsimage)
    except Exception:
        pass

def classification_state_from_name(name):
    normalized = name.lower()
    if normalized == "prepared_circle" or "prepared" in normalized:
        return "prepared"
    if normalized == "circle" or "finished" in normalized:
        return "finished"
    return "unknown"

def write_classification_status(
    symbol_name,
    score,
    state=None,
    source="draw",
    details=None,
    image_path="test.png",
    rotation=None,
    direction=None
):
    if state is None:
        state = classification_state_from_name(symbol_name)
    payload = {
        "symbol_name": symbol_name,
        "score": score,
        "state": state,
        "source": source,
        "details": details or "",
        "image_path": image_path,
        "rotation": rotation,
        "direction": direction,
        "timestamp": datetime.now().isoformat()
    }
    try:
        with SHARED_STATUS_FILE.open("w") as f:
            json.dump(payload, f, indent=2)
    except Exception as exc:
        print(f"Failed to write classification status: {exc}")
    return payload

def read_classification_status():
    if not SHARED_STATUS_FILE.exists():
        return {}
    try:
        with SHARED_STATUS_FILE.open() as f:
            return json.load(f)
    except Exception as exc:
        print(f"Failed to read classification status: {exc}")
        return {}

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

def get_contour_from_array(img_array):
    """Accept a NumPy image array (RGB or BGR) and return contours.
    This avoids saving the image to disk when the caller already has
    the image in memory (e.g. from a PIL Image).
    """
    if img_array is None:
        return []
    # Convert to numpy array if PIL Image was provided
    try:
        import numpy as _np
        img = _np.array(img_array)
    except Exception:
        img = img_array
    # If image has 3 channels assume RGB and convert to grayscale
    if len(img.shape) == 3 and img.shape[2] == 3:
        # detect whether it's RGB (PIL) or BGR (cv)
        # We'll try converting from RGB first then threshold
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
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
    def hu_distance(c1, c2):
        m1 = cv2.moments(c1)
        m2 = cv2.moments(c2)
        h1 = cv2.HuMoments(m1).flatten()
        h2 = cv2.HuMoments(m2).flatten()
        h1 = -np.sign(h1) * np.log10(np.abs(h1) + 1e-30)
        h2 = -np.sign(h2) * np.log10(np.abs(h2) + 1e-30)
        return float(np.sum(np.abs(h1 - h2)))

    def bbox_ratio_diff(c1, c2):
        x1, y1, w1, h1 = cv2.boundingRect(c1)
        x2, y2, w2, h2 = cv2.boundingRect(c2)
        r1 = w1 / (h1 if h1 != 0 else 1)
        r2 = w2 / (h2 if h2 != 0 else 1)
        return abs(r1 - r2)
    scores = []
    for name, templates in library.items():
        best_score = float("inf")
        for template in templates:
            shape_score = cv2.matchShapes(
                contour,
                template,
                cv2.CONTOURS_MATCH_I1,
                0
            )
            hu_score = hu_distance(contour, template)
            br = bbox_ratio_diff(contour, template)
            # combined score: prioritize matchShapes, include Hu and bbox ratio
            combined = (
                0.6 * shape_score
                + 0.3 * (hu_score / (1.0 + hu_score))
                + 0.1 * br
            )
            best_score = min(best_score, combined)
        scores.append((name, best_score))
    scores.sort(key=lambda x: x[1])
    best_name, best_score = scores[0]
    return best_name, best_score

def get_top_matches(
    contour,
    library,
    limit=5
):
    def hu_distance(c1, c2):
        m1 = cv2.moments(c1)
        m2 = cv2.moments(c2)
        h1 = cv2.HuMoments(m1).flatten()
        h2 = cv2.HuMoments(m2).flatten()
        # use log transform to compress dynamic range
        h1 = -np.sign(h1) * np.log10(np.abs(h1) + 1e-30)
        h2 = -np.sign(h2) * np.log10(np.abs(h2) + 1e-30)
        return float(np.sum(np.abs(h1 - h2)))
    scores = []
    for name, templates in library.items():
        best_score = float("inf")
        for template in templates:
            shape_score = cv2.matchShapes(
                contour,
                template,
                cv2.CONTOURS_MATCH_I1,
                0
            )
            hu_score = hu_distance(contour, template)
            # bbox ratio diff
            x1, y1, w1, h1 = cv2.boundingRect(contour)
            x2, y2, w2, h2 = cv2.boundingRect(template)
            r1 = w1 / (h1 if h1 != 0 else 1)
            r2 = w2 / (h2 if h2 != 0 else 1)
            br = abs(r1 - r2)
            # combined score: prioritize matchShapes, include Hu and bbox ratio
            combined = (
                0.6 * shape_score
                + 0.3 * (hu_score / (1.0 + hu_score))
                + 0.1 * br
            )
            best_score = min(
                best_score,
                combined
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
    # determine rotation/direction for directional symbols
    directional_symbols = {
        "arrow",
        "wavy_arrow",
        "wavyarrow",
        "pull",
        "push",
        "column",
        "crush",
        "levitation",
        "spiral"
    }
    rotation = None
    try:
        lname = name.lower()
        if any(ds in lname for ds in directional_symbols):
            if "arrow" in lname:
                rotation = get_arrow_rotation(contour)
            else:
                # use principal axis via moments / PCA
                pts = contour.reshape(-1, 2).astype(float)
                mean = pts.mean(axis=0)
                cov = np.cov((pts - mean).T)
                try:
                    vals, vecs = np.linalg.eig(cov)
                    principal = vecs[:, np.argmax(vals)]
                    rotation = np.degrees(np.arctan2(principal[1], principal[0]))
                    rotation = (rotation + 360) % 360
                except Exception:
                    rotation = None
    except Exception:
        rotation = None
    return name, score, rotation

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
