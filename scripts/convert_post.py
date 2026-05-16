"""Convert a Google Docs .docx export into a blog post markdown file.

Pipeline:
  1. pandoc reads the .docx and converts OMML equations to LaTeX (`$...$`).
  2. Images embedded in the docx are extracted to public/posts/<slug>/.
  3. Image paths in the markdown are rewritten to absolute web paths.
  4. A small set of stylistic substitutions are applied inside math regions
     (Google Docs can't express \\mathbb{E} or \\mathrm{KL} semantically).
  5. Frontmatter is prepended.
  6. The result is written to src/content/blog/<slug>.md.

Usage:
    python scripts/convert_post.py raw_posts/post.docx \\
        --slug from-double-descent-to-scaling-laws \\
        --title "From Double Descent to Scaling Laws" \\
        --date 2026-05-16 \\
        --description "Reconciling 2010s overfitting with 2020s LLM scaling."

If --slug/--title/--date are omitted, defaults are derived from the filename
and today's date. Re-run freely; output files are overwritten.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CONTENT_DIR = REPO / "src" / "content" / "blog"
PUBLIC_POSTS_DIR = REPO / "public" / "posts"


def slugify(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", name).strip("-").lower()
    return s or "post"


def run_pandoc(docx: Path, media_dir: Path) -> str:
    """Convert .docx to markdown with LaTeX math, extracting media to media_dir."""
    media_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "pandoc",
            str(docx),
            "-f", "docx",
            "-t", "gfm+tex_math_dollars",
            "--wrap=none",
            "--extract-media", str(media_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def rewrite_image_paths(md: str, slug: str) -> str:
    """Pandoc emits absolute fs paths for extracted media; rewrite to /posts/<slug>/..."""
    # Pandoc image syntax: ![alt](path) or ![alt](path "title")
    # We rewrite anything under PUBLIC_POSTS_DIR / slug to a web-absolute path.
    public_prefix = str(PUBLIC_POSTS_DIR / slug)

    def repl(m: re.Match[str]) -> str:
        alt, path = m.group(1), m.group(2)
        if public_prefix in path:
            web_path = path.split(public_prefix, 1)[1]
            return f"![{alt}](/posts/{slug}{web_path})"
        return m.group(0)

    return re.sub(r"!\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)", repl, md)


# Substitutions applied INSIDE math regions only. Google Docs can render
# blackboard E, parallel bars, etc., but its OMML export collapses them to
# plain glyphs that pandoc can't recover semantically.
MATH_SUBS: list[tuple[str, str]] = [
    # Common operators that look like regular letters in Google Docs.
    # Convert standalone E_X / E_D patterns to \mathbb{E}.
    (r"(?<![A-Za-z\\])E_([A-Za-z])", r"\\mathbb{E}_\1"),
    (r"(?<![A-Za-z\\])E_\{([^}]+)\}", r"\\mathbb{E}_{\1}"),
    # KL divergence label.
    (r"(?<![A-Za-z\\])KL\(", r"\\mathrm{KL}("),
    # Parallel bars: || -> \| (KaTeX renders \| as ∥).
    (r"\|\|", r"\\|"),
]


def apply_math_substitutions(md: str) -> str:
    """Apply MATH_SUBS only inside $...$ and $$...$$ regions."""
    pattern = re.compile(r"(\$\$.+?\$\$|\$[^$\n]+?\$)", re.DOTALL)

    def process(match: re.Match[str]) -> str:
        s = match.group(0)
        for pat, repl in MATH_SUBS:
            s = re.sub(pat, repl, s)
        return s

    return pattern.sub(process, md)


def build_frontmatter(title: str, description: str, date: str) -> str:
    desc = description.replace('"', "'")
    return (
        "---\n"
        f"title: \"{title}\"\n"
        f"description: \"{desc}\"\n"
        f"pubDate: \"{date}\"\n"
        "---\n\n"
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("docx", type=Path, help="Path to the .docx file (typically under raw_posts/).")
    p.add_argument("--slug", default=None, help="URL slug (default: derived from filename).")
    p.add_argument("--title", default=None, help="Post title (default: derived from filename).")
    p.add_argument("--date", default=None, help="Publication date YYYY-MM-DD (default: today).")
    p.add_argument("--description", default="", help="Short description for SEO + post list.")
    args = p.parse_args()

    if not args.docx.exists():
        print(f"error: {args.docx} not found", file=sys.stderr)
        return 1

    title = args.title or args.docx.stem
    slug = args.slug or slugify(title)
    date = args.date or dt.date.today().isoformat()

    media_dir = PUBLIC_POSTS_DIR / slug
    if media_dir.exists():
        shutil.rmtree(media_dir)

    print(f"slug:   {slug}")
    print(f"title:  {title}")
    print(f"date:   {date}")
    print(f"media:  {media_dir}")

    md = run_pandoc(args.docx, media_dir)
    md = rewrite_image_paths(md, slug)
    md = apply_math_substitutions(md)
    md = build_frontmatter(title, args.description, date) + md

    CONTENT_DIR.mkdir(parents=True, exist_ok=True)
    out = CONTENT_DIR / f"{slug}.md"
    out.write_text(md, encoding="utf-8")
    print(f"wrote:  {out}")
    if media_dir.exists() and any(media_dir.iterdir()):
        n = sum(1 for _ in media_dir.rglob("*") if _.is_file())
        print(f"images: {n} extracted to {media_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
