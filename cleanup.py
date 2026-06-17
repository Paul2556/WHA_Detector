import json
from pathlib import Path
from collections import Counter
import main
SYMBOLS_DIR = "symbols"
library = main.load_symbol_library()
for symbol_dir in Path(SYMBOLS_DIR).iterdir():
    if not symbol_dir.is_dir():
        continue
    pngs = list(
        symbol_dir.glob("*.png")
    )
    if not pngs:
        continue
    example = pngs[0]
    try:
        contours = main.get_contour(
            example
        )
        components = []
        for contour in contours:
            try:
                name, score = (
                    main.classify(
                        contour,
                        library
                    )
                )
                components.append(
                    name
                )
            except:
                pass
        metadata = {
            "name": symbol_dir.name,
            "components": dict(
                Counter(
                    components
                )
            ),
            "examples": len(
                pngs
            )
        }
        metadata_file = (
            symbol_dir /
            "metadata.json"
        )
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
            f"Updated: "
            f"{symbol_dir.name}"
        )
    except Exception as e:
        print(
            f"Failed: "
            f"{symbol_dir.name}"
        )
        print(e)
