"""Stage the paper and original video for the static project page and check its links."""
from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
import shutil
from urllib.parse import urlparse, unquote

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "website/dist"


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.refs = []
        self.ids = []
        self.headings = 0
        self.images = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.append(attrs["id"])
        if tag == "h1":
            self.headings += 1
        if tag == "img":
            self.images.append(attrs)
        for name in ("src", "href", "poster", "data-zoom"):
            if name in attrs:
                self.refs.append(attrs[name])


def main():
    assets = DIST / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    for source, target in [(ROOT / "paper/26-2531_03_MS.pdf", assets / "paper.pdf"),
                           (ROOT / "paper/Video_Final.mp4", assets / "overview.mp4")]:
        shutil.copy2(source, target)
    page = Page()
    page.feed((DIST / "index.html").read_text())
    assert page.headings == 1, "The page must have one main heading"
    assert len(page.ids) == len(set(page.ids)), "Duplicate HTML IDs"
    assert all(image.get("alt") for image in page.images if image.get("id") != "dialog-image"), "Missing image descriptions"
    for ref in page.refs:
        url = urlparse(ref)
        if url.scheme or url.netloc:
            continue
        if url.path:
            target = DIST / unquote(url.path)
            assert target.is_file(), f"Missing local asset: {ref}"
        elif url.fragment:
            assert url.fragment in page.ids or url.fragment.startswith("i-"), f"Unknown anchor: {ref}"
    print(f"Static page ready: {len(page.refs)} links/assets checked; original PDF/video staged.")


if __name__ == "__main__":
    main()
