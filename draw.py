import tkinter as tk
from PIL import Image, ImageDraw
import cv2

import main

CANVAS_WIDTH = 600
CANVAS_HEIGHT = 400
BRUSH_SIZE = 8

library = main.load_symbol_library()


class DrawingCanvas:

    def __init__(self, root):
        self.root = root

        self.dark_mode = False

        tk.Button(
            root,
            text="Classify Override",
            command=self.classify_override
        ).pack()

        tk.Button(
            root,
            text="Toggle Dark Mode",
            command=self.toggle_dark_mode
        ).pack()

        self.canvas = tk.Canvas(
            root,
            width=CANVAS_WIDTH,
            height=CANVAS_HEIGHT,
            bg="white"
        )

        self.canvas.pack()

        self.result_label = tk.Label(
            root,
            text="Draw a symbol..."
        )

        self.result_label.pack()

        self.image = Image.new(
            "RGB",
            (CANVAS_WIDTH, CANVAS_HEIGHT),
            "white"
        )

        self.draw = ImageDraw.Draw(
            self.image
        )

        self.canvas.bind(
            "<Button-1>",
            self.start_draw
        )

        self.canvas.bind(
            "<B1-Motion>",
            self.draw_line
        )

        self.canvas.bind(
            "<Button-3>",
            self.start_erase
        )

        self.canvas.bind(
            "<B3-Motion>",
            self.erase_line
        )

        self.last_x = None
        self.last_y = None

        tk.Button(
            root,
            text="Save / Classify",
            command=self.save_image
        ).pack()

        tk.Button(
            root,
            text="Clear",
            command=self.clear_canvas
        ).pack()

    def classify_override(self):

        global library

        self.image.save(
            "test.png"
        )

        try:

            contours = main.get_contour(
                "test.png"
            )

            popup = tk.Toplevel(
                self.root
            )

            popup.title(
                "Override Classification"
            )

            tk.Label(
                popup,
                text="Enter symbol name:"
            ).pack()

            entry = tk.Entry(
                popup
            )

            entry.pack()

            entry.focus()

            def save_override():

                global library

                symbol_name = (
                    entry.get()
                    .strip()
                )

                if not symbol_name:
                    return

                main.save_symbol_example(
                    "test.png",
                    symbol_name
                )

                components = []

                for contour in contours:

                    try:

                        component_name, _ = (
                            main.classify(
                                contour,
                                library
                            )
                        )

                        components.append(
                            component_name
                        )

                    except:
                        pass

                main.update_symbol_metadata(
                    symbol_name,
                    components
                )

                library = (
                    main.load_symbol_library()
                )

                self.result_label.config(
                    text=(
                        f"Saved as "
                        f"'{symbol_name}'"
                    )
                )

                popup.destroy()

            tk.Button(
                popup,
                text="Save Override",
                command=save_override
            ).pack()

            entry.bind(
                "<Return>",
                lambda e: save_override()
            )

        except Exception as e:

            print(e)
    def toggle_dark_mode(self):

        self.dark_mode = not self.dark_mode

        if self.dark_mode:

            self.root.configure(
                bg="#1e1e1e"
            )

            self.canvas.configure(
                bg="#2d2d2d"
            )

            self.result_label.configure(
                bg="#1e1e1e",
                fg="white"
            )

        else:

            self.root.configure(
                bg="SystemButtonFace"
            )

            self.canvas.configure(
                bg="white"
            )

            self.result_label.configure(
                bg="SystemButtonFace",
                fg="black"
            )

    def start_draw(self, event):
        self.last_x = event.x
        self.last_y = event.y

    def start_erase(self, event):
        self.last_x = event.x
        self.last_y = event.y

    def draw_line(self, event):

        self.canvas.create_line(
            self.last_x,
            self.last_y,
            event.x,
            event.y,
            fill="black",
            width=BRUSH_SIZE,
            capstyle=tk.ROUND,
            smooth=True
        )

        self.draw.line(
            [
                (self.last_x, self.last_y),
                (event.x, event.y)
            ],
            fill="black",
            width=BRUSH_SIZE
        )

        self.last_x = event.x
        self.last_y = event.y

        self.auto_classify()

    def erase_line(self, event):

        self.canvas.create_line(
            self.last_x,
            self.last_y,
            event.x,
            event.y,
            fill="white",
            width=BRUSH_SIZE * 2,
            capstyle=tk.ROUND,
            smooth=True
        )

        self.draw.line(
            [
                (self.last_x, self.last_y),
                (event.x, event.y)
            ],
            fill="white",
            width=BRUSH_SIZE * 2
        )

        self.last_x = event.x
        self.last_y = event.y

        self.auto_classify()

    def clear_canvas(self):

        self.canvas.delete("all")

        self.image = Image.new(
            "RGB",
            (CANVAS_WIDTH, CANVAS_HEIGHT),
            "white"
        )

        self.draw = ImageDraw.Draw(
            self.image
        )

        self.result_label.config(
            text="Draw a symbol..."
        )

    def save_image(self):

        global library

        self.image.save(
            "test.png"
        )

        try:

            contours = main.get_contour(
                "test.png"
            )

            contour = max(
                contours,
                key=cv2.contourArea
            )

            name, score = main.classify(
                contour,
                library
            )

            rotation = None

            if name == "arrow":

                rotation = main.get_arrow_rotation(
                    contour
                )

                main.debug_arrow_tip(contour)

            top_matches = main.get_top_matches(
                contour,
                library,
                5
            )

            results = (
                f"{name}\n"
                f"Score: {score:.4f}\n"
            )

            if rotation is not None:

                direction = main.rotation_to_direction(
                    rotation
                )

                results += (
                    f"Rotation: "
                    f"{rotation:.1f}° "
                    f"{direction}\n"
                )

            else:

                results += (
                    "Rotation: N/A\n"
                )

            results += "\n"

            for symbol, symbol_score in top_matches:

                results += (
                    f"{symbol}: "
                    f"{symbol_score:.4f}\n"
                )

            self.result_label.config(
                text=results
            )

            print()
            print("Top Matches")
            print("-" * 40)

            for symbol, symbol_score in top_matches:

                print(
                    f"{symbol:<20}"
                    f"{symbol_score:.6f}"
                )

            print("-" * 40)

            print(
                f"Best Match : {name}"
            )

            print(
                f"Score      : {score:.6f}"
            )

            if score <= main.UNKNOWN_THRESHOLD:

                main.save_symbol_example(
                    "test.png",
                    name
                )

                library = main.load_symbol_library()

                print(
                    f"Added example "
                    f"to {name}"
                )

            else:

                self.ask_for_symbol_name(
                    default_name=name
                )

        except Exception as e:

            self.result_label.config(
                text=f"Error: {e}"
            )

            print(e)

    def ask_for_symbol_name(
        self,
        default_name=""
    ):

        popup = tk.Toplevel(
            self.root
        )

        popup.title(
            "New Symbol"
        )

        tk.Label(
            popup,
            text="Enter symbol name:"
        ).pack()

        entry = tk.Entry(
            popup
        )

        entry.insert(
            0,
            default_name
        )

        entry.pack()

        entry.focus()

        def save_symbol():

            global library

            symbol_name = (
                entry.get()
                .strip()
            )

            if not symbol_name:
                return

            main.save_symbol_example(
                "test.png",
                symbol_name
            )

            contours = main.get_contour(
                "test.png"
            )

            components = []

            for contour in contours:

                component_name, _ = (
                    main.classify(
                        contour,
                        library
                    )
                )

                components.append(
                    component_name
                )

            main.update_symbol_metadata(
                symbol_name,
                components
            )
            library = main.load_symbol_library()

            self.result_label.config(
                text=(
                    f"Added "
                    f"'{symbol_name}'"
                )
            )

            popup.destroy()

        tk.Button(
            popup,
            text="Save Symbol",
            command=save_symbol
        ).pack()

        entry.bind(
            "<Return>",
            lambda e: save_symbol()
        )

    def auto_classify(self):

        global library

        try:

            self.image.save(
                "test.png"
            )

            contours = main.get_contour(
                "test.png"
            )

            contour = max(
                contours,
                key=cv2.contourArea
            )

            name, score = main.classify(
                contour,
                library
            )

            rotation = None

            if name == "arrow":
                rotation = main.get_arrow_rotation(
                    contour
                )

            top_matches = main.get_top_matches(
                contour,
                library,
                5
            )

            results = (
                f"{name}\n"
                f"Score: {score:.4f}\n"
            )

            if rotation is not None:

                direction = main.rotation_to_direction(
                    rotation
                )

                results += (
                    f"Rotation: "
                    f"{rotation:.1f}° "
                    f"{direction}\n"
                )

            else:

                results += (
                    "Rotation: N/A\n"
                )

            results += "\n"

            for symbol, symbol_score in top_matches:

                results += (
                    f"{symbol}: "
                    f"{symbol_score:.4f}\n"
                )

            self.result_label.config(
                text=results
            )

        except:
            pass


root = tk.Tk()

root.title(
    "WHA Glyph Trainer"
)

DrawingCanvas(
    root
)

root.mainloop()