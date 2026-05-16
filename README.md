# jl-blog

Personal blog. Static HTML built with [Astro](https://astro.build), math via [KaTeX](https://katex.org), code highlighting via [Shiki](https://shiki.style). Deployed to Cloudflare Pages.

## Local development

```bash
npm install
npm run dev        # http://localhost:4321
npm run build      # production build to ./dist
npm run preview    # serve the built ./dist locally
```

## Writing a post

Plain Markdown files in `src/content/blog/<slug>.md`. Frontmatter:

```yaml
---
title: "..."
description: "..."
pubDate: "2026-05-16"
updatedDate: "2026-05-17"   # optional
---
```

Math: `$inline$` and `$$display$$`. Code: standard triple-backtick fences (` ```python `, etc.).

### From a Google Doc

The Markdown export from Google Docs flattens equations to plain text. Use the conversion script instead — it goes via DOCX, where equations are preserved as OMML and Pandoc recovers them as LaTeX.

```bash
# 1) In Google Docs: File -> Download -> Microsoft Word (.docx)
# 2) Drop the .docx into ./raw_posts/ (gitignored).
# 3) Run:
python scripts/convert_post.py raw_posts/post.docx \
    --slug my-post \
    --title "My Post" \
    --date 2026-05-16 \
    --description "Optional one-liner."
```

The script:
- Runs Pandoc with `gfm+tex_math_dollars` (OMML → LaTeX math).
- Extracts embedded images to `public/posts/<slug>/` and rewrites paths.
- Applies a small set of stylistic substitutions inside math regions (`E_X` → `\mathbb{E}_X`, `KL(` → `\mathrm{KL}(`, `||` → `\|`).
- Writes `src/content/blog/<slug>.md`.

Re-run freely — the output is overwritten. Iterate on `scripts/convert_post.py` if you hit edge cases (e.g. additional math substitutions). For equations Pandoc can't recover, screenshot from the live Google Doc and use [Mathpix Snipping Tool](https://mathpix.com/snipping-tool).

Pandoc requirement: `brew install pandoc` (already installed on the laptop). Python 3.9+ (uses only the standard library).

## Deploying to Cloudflare Pages

One-time setup:

1. Push this repo to GitHub (private is fine).
2. Sign in to https://dash.cloudflare.com (free, no card needed).
3. **Workers & Pages → Create → Pages → Connect to Git**.
4. Authorize the Cloudflare GitHub app, select this repo.
5. Build configuration:
   - **Framework preset**: Astro
   - **Build command**: `npm run build`
   - **Build output directory**: `dist`
   - **Node version**: set environment variable `NODE_VERSION = 22`
6. Save and Deploy. First build takes ~1 min.

Every push to `main` triggers a production deploy. Other branches get preview URLs at `<branch>.<project>.pages.dev`.

The default public URL is `https://<project-name>.pages.dev`. If using a custom domain, add it under **Custom domains** in the Pages project.

If you change the canonical site URL, update `site:` in `astro.config.mjs`.

## Layout

```
src/
  components/   small UI pieces (header, footer, date)
  layouts/      BlogPost.astro — wraps every post
  pages/        routes (index, blog/, about, rss)
  content/      blog content (markdown files)
  styles/       global.css
public/         static assets (favicon, post images)
scripts/        convert_post.py
raw_posts/      (gitignored) drop .docx here for conversion
```
