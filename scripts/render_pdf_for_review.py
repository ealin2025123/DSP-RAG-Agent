"""Render an image-only PDF into temporary PNGs for manual coverage review."""

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "vendor"))
import pypdfium2 as pdfium  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf")
    parser.add_argument("output")
    parser.add_argument("--scale", type=float, default=1.6)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    document = pdfium.PdfDocument(args.pdf)
    for index, page in enumerate(document, 1):
        image = page.render(scale=args.scale).to_pil()
        image.save(str(output / "page_{:02d}.png".format(index)))
    print("Rendered {} pages to {}".format(len(document), output))


if __name__ == "__main__":
    main()
