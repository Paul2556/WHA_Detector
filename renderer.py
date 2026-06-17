import os
import sys
import tkinter as tk
from enum import Enum

import cv2
import main


DEFAULT_IMAGE_PATH = "test.png"
REFRESH_MS = 500


class CircleState(Enum):
    PREPARED = "prepared"
    FINISHED = "finished"
    UNKNOWN = "unknown"


class RendererWindow:
    WINDOW_SIZE = 300

    def __init__(self, image_path: str = DEFAULT_IMAGE_PATH):
        self.image_path = image_path
        self.state = CircleState.UNKNOWN
        self.library = main.load_symbol_library()

        self.root = tk.Tk()
        main.set_app_icon(self.root)
        self.root.title("Circle Renderer")
        self.root.geometry(f"{self.WINDOW_SIZE}x{self.WINDOW_SIZE + 120}")
        self.root.resizable(False, False)

        self.canvas = tk.Canvas(
            self.root,
            width=self.WINDOW_SIZE,
            height=self.WINDOW_SIZE,
            highlightthickness=0
        )
        self.canvas.pack()

        self.status_label = tk.Label(
            self.root,
            text="",
            font=("Arial", 14)
        )
        self.status_label.pack(pady=6)

        self.detail_label = tk.Label(
            self.root,
            text="",
            font=("Arial", 12)
        )
        self.detail_label.pack(pady=2)

        self.refresh_button = tk.Button(
            self.root,
            text="Refresh",
            command=self.render_state
        )
        self.refresh_button.pack(pady=6)

        self.update_loop()

    def classify_state(self):
        shared_status = main.read_classification_status()

        if shared_status:
            state = shared_status.get("state", "unknown")
            symbol_name = shared_status.get("symbol_name", "unknown")
            score = shared_status.get("score", 0)
            source = shared_status.get("source", "shared_status")
            details = shared_status.get("details", "")

            if state == "prepared":
                return CircleState.PREPARED, (
                    f"Shared: {symbol_name} ({score:.4f}) from {source}. {details}"
                )

            if state == "finished":
                return CircleState.FINISHED, (
                    f"Shared: {symbol_name} ({score:.4f}) from {source}. {details}"
                )

            return CircleState.UNKNOWN, (
                f"Shared: {symbol_name} ({score:.4f}) from {source}. {details}"
            )

        if not self.library:
            return CircleState.UNKNOWN, "No symbols loaded"

        if not os.path.exists(self.image_path):
            return CircleState.UNKNOWN, f"Waiting for {self.image_path}"

        try:
            contours = main.get_contour(self.image_path)
        except Exception as exc:
            return CircleState.UNKNOWN, f"Error reading image: {exc}"

        if not contours:
            return CircleState.UNKNOWN, "No contours found"

        contour = max(contours, key=cv2.contourArea)
        name, score = main.classify(contour, self.library)

        if name == "prepared_circle" or "prepared" in name:
            return CircleState.PREPARED, f"Detected: {name} ({score:.4f})"

        if name == "circle" or "finished" in name:
            return CircleState.FINISHED, f"Detected: {name} ({score:.4f})"

        return CircleState.UNKNOWN, f"Detected: {name} ({score:.4f})"

    def render_state(self):
        self.state, detail = self.classify_state()

        if self.state == CircleState.PREPARED:
            fill_color = "white"
            text_color = "black"
            status_text = "Prepared"
        elif self.state == CircleState.FINISHED:
            fill_color = "black"
            text_color = "white"
            status_text = "Finished"
        else:
            fill_color = "gray"
            text_color = "white"
            status_text = "Unknown"

        self.canvas.delete("all")
        self.canvas.create_rectangle(
            0,
            0,
            self.WINDOW_SIZE,
            self.WINDOW_SIZE,
            fill=fill_color,
            outline=fill_color
        )
        self.canvas.create_text(
            self.WINDOW_SIZE // 2,
            self.WINDOW_SIZE // 2,
            text=status_text,
            fill=text_color,
            font=("Arial", 28, "bold")
        )

        self.status_label.config(
            text=f"Circle state: {status_text}"
        )
        self.detail_label.config(
            text=detail
        )

    def update_loop(self):
        self.render_state()
        self.root.after(REFRESH_MS, self.update_loop)

    def run(self):
        self.root.mainloop()


def parse_image_path() -> str:
    if len(sys.argv) < 2:
        return DEFAULT_IMAGE_PATH

    return sys.argv[1]


if __name__ == "__main__":
    image_path = parse_image_path()
    renderer = RendererWindow(image_path=image_path)
    renderer.run()
