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
            # `markdown` (pandoc's own) emits plain `$...$` for math; `gfm` wraps
            # the math content in backticks which remark-math won't parse.
            # `pipe_tables` (with simple/multiline/grid disabled) gives
            # GFM `| col | col |` tables that remark-gfm understands.
            "-t",
            "markdown+tex_math_dollars-raw_attribute"
            "+pipe_tables-multiline_tables-simple_tables-grid_tables",
            "--wrap=none",
            "--extract-media", str(media_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def rewrite_image_paths(md: str, slug: str) -> str:
    """Convert all image refs to markdown syntax with web-absolute paths.

    Pandoc emits both markdown `![alt](path)` and raw `<img src="path" .../>`
    tags (the latter when the source has attributes like width/style). We
    normalize both to plain `![alt](/posts/<slug>/...)`.
    """
    public_prefix = str(PUBLIC_POSTS_DIR / slug)

    def to_web_path(fs_path: str) -> str | None:
        if public_prefix in fs_path:
            return f"/posts/{slug}{fs_path.split(public_prefix, 1)[1]}"
        return None

    def md_repl(m: re.Match[str]) -> str:
        alt, path = m.group(1), m.group(2)
        web = to_web_path(path)
        return f"![{alt}]({web})" if web else m.group(0)

    md = re.sub(r"!\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)", md_repl, md)

    def img_tag_repl(m: re.Match[str]) -> str:
        src = m.group(1)
        web = to_web_path(src)
        return f"![]({web})" if web else m.group(0)

    md = re.sub(r"<img\s+[^>]*src=\"([^\"]+)\"[^>]*/?>", img_tag_repl, md)
    return md


def strip_html_artifacts(md: str) -> str:
    """Remove Google Docs styling artifacts and pandoc-specific syntax that
    plain markdown renderers won't understand."""
    # Raw underline tags from Google Docs underlined links.
    md = re.sub(r"</?u>", "", md)

    # Pandoc image-attribute blocks: ![](path){width="..." ...}  ->  ![](path)
    md = re.sub(r"(\!\[[^\]]*\]\([^)]+\))\{[^}]*\}", r"\1", md)

    # Google Docs underlined citation links come through as:
    #   [[\[Author, year\]]{.underline}](url)
    # Collapse to a plain link: [Author, year](url).
    md = re.sub(
        r"\[\[\\\[([^\]]+?)\\\]\]\{\.underline\}\]\(([^)]+)\)",
        r"[\1](\2)",
        md,
    )
    # Any remaining inline-attribute spans (other underlines, smallcaps, etc.).
    md = re.sub(r"\]\{[^}]*\}", "]", md)
    # Google Docs in-page bookmark links written with bracket styling come
    # through as `[**[Section name]**](#anchor)` after the underline strip
    # above — collapse to `[**Section name**](#anchor)`.
    md = re.sub(
        r"\[\*\*\[([^\]]+?)\]\*\*\]\((#[^)]+)\)",
        r"[**\1**](\2)",
        md,
    )
    for _ in range(3):
        new = re.sub(r"\[\[([^\[\]]+)\]\]\(", r"[\1](", md)
        if new == md:
            break
        md = new

    # Unescape characters pandoc escaped that don't need to be escaped in
    # plain markdown. Brackets in math are `\lbrack`/`\rbrack`, never `\[`,
    # so global unescape is safe.
    md = re.sub(r"\\\[", "[", md)
    md = re.sub(r"\\\]", "]", md)
    md = re.sub(r"\\'", "'", md)
    md = re.sub(r"-\\>", "->", md)

    # Repair math regions where pandoc escaped the closing `$` delimiter,
    # e.g. `$ ... \rbrack\$` -> `$ ... \rbrack$`. Heuristic: a line that
    # opens with `$` and ends with `\$` is a malformed inline math close.
    md = re.sub(r"^\$(.+?)\\\$$", r"$\1$", md, flags=re.MULTILINE)

    # Promote standalone-line inline math to display math.
    # Pandoc emits all OMML equations as `$...$` (inline), even when they
    # occupy a whole paragraph. Remark-math only treats `$$` as display
    # math when the delimiters sit on their own lines, so we expand to:
    #   $$
    #   <content>
    #   $$
    # This makes KaTeX emit `<span class="katex-display">`, which our CSS
    # extends wider than the prose column.
    md = re.sub(r"^\$([^$\n]+)\$$", r"$$\n\1\n$$", md, flags=re.MULTILINE)

    # Place inline images on their own line so they don't sit at the end
    # of a paragraph.
    md = re.sub(r"([^\n])(\!\[[^\]]*\]\([^)]+\))", r"\1\n\n\2", md)

    return md


# Substitutions applied INSIDE math regions only. Google Docs can render
# blackboard E, parallel bars, etc., but its OMML export collapses them to
# plain glyphs that pandoc can't recover semantically.
_GREEK = (
    "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu "
    "nu xi pi rho sigma tau upsilon phi chi psi omega "
    "Gamma Delta Theta Lambda Xi Pi Sigma Upsilon Phi Psi Omega"
).split()
_LATEX_CMDS = _GREEK + [
    "mathbb", "mathrm", "mathcal", "mathbf", "mathsf", "mathit",
    "sum", "prod", "int", "frac", "sqrt", "log", "exp", "sin", "cos",
    "le", "ge", "leq", "geq", "neq", "approx", "sim", "to", "infty",
    "left", "right", "cdot", "times", "div", "pm", "mp",
    "partial", "nabla", "in", "notin", "subset", "supset",
    "forall", "exists", "implies", "iff", "vec", "hat", "bar", "tilde",
    "begin", "end", "text", "label",
]

# Substitutions applied INSIDE math regions only.
#
# Google Docs renders \theta etc. as plain text characters (not via the
# equation editor's symbol palette), so pandoc's OMML reader emits them as
# `\backslash theta`. We rebuild the LaTeX command. We also handle a few
# semantic gaps (E_X -> \mathbb{E}_X, KL -> \mathrm{KL}, || -> \|).
# LaTeX function names that should render upright (non-italic) in math mode.
# Google Docs writes these as plain letters; we prefix them with a backslash
# so KaTeX renders them as operators.
_MATH_FUNCS = [
    "log", "ln", "exp", "sin", "cos", "tan",
    "det", "lim", "max", "min", "sup", "inf", "arg", "Pr",
]

MATH_SUBS: list[tuple[str, str]] = [
    # Unescape LaTeX commands the OMML reader serialized as plain text.
    (rf"\\backslash\s+({'|'.join(_LATEX_CMDS)})\b", r"\\\1"),
    # Strip pandoc's `\ ` (escaped space) inside math.
    (r"\\ ", " "),
    # Blackboard E for expectations.
    (r"(?<![A-Za-z\\])E_([A-Za-z])", r"\\mathbb{E}_\1"),
    (r"(?<![A-Za-z\\])E_\{([^}]+)\}", r"\\mathbb{E}_{\1}"),
    # KL divergence label.
    (r"(?<![A-Za-z\\])KL\(", r"\\mathrm{KL}("),
    # Parallel bars: || -> \| (KaTeX renders \| as ∥).
    (r"\|\|", r"\\|"),
    # Common math function names: log -> \log etc.
    (rf"(?<![\\A-Za-z])({'|'.join(_MATH_FUNCS)})\b", r"\\\1"),
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
    md = strip_html_artifacts(md)
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
