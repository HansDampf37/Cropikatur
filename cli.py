import os
from pathlib import Path

from cropikatur import crop_image
from debugging import ImageDebugger, NullImageDebugger


def main() -> None:
    """
    CLI entry point for cropping documents in images.
    Handles both single files and folders, resolving output paths intelligently.
    """
    import argparse
    parser = argparse.ArgumentParser(description="Crop image(s) to detected document edges.")
    parser.add_argument("input", help="Path to input image or folder")
    parser.add_argument("output", nargs="?", help="Optional output file or folder path")
    parser.add_argument("--debug", action="store_true", help="Show debug images")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_arg = Path(args.output).resolve() if args.output else None
    debug_images = ImageDebugger() if args.debug else NullImageDebugger()

    def is_probably_file(path: Path) -> bool:
        """Returns True if the path looks like a file (has an extension)."""
        return path.suffix != ""

    if not input_path.exists():
        print(f"Error: Input path '{input_path}' does not exist.")
        return

    if input_path.is_file():
        if output_arg is None:
            output_path = input_path.with_stem(input_path.stem + "_cropped")
        elif output_arg.exists() and output_arg.is_dir():
            output_path = output_arg / input_path.name
        else:
            output_path = (
                output_arg / input_path.name
                if not output_arg.exists() and not is_probably_file(output_arg)
                else output_arg
            )

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        crop_image(str(input_path), str(output_path), imageDebugger=debug_images)

    elif input_path.is_dir():
        if output_arg is None:
            output_dir = input_path.parent / (input_path.name + "_cropped")
        elif output_arg.exists() and output_arg.is_file() or not output_arg.exists() and is_probably_file(output_arg):
            print("Error: Cannot write multiple files to a single file.")
            return
        else:
            output_dir = output_arg

        output_dir.mkdir(parents=True, exist_ok=True)

        for filename in os.listdir(input_path):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                print(f"⚠Skipping unsupported file: {filename}")
                continue

            input_file = input_path / filename
            output_file = output_dir / filename
            crop_image(str(input_file), str(output_file), imageDebugger=debug_images)


if __name__ == "__main__":
    main()