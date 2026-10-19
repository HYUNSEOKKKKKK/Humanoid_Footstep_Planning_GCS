# Paper project page

Public URL: <https://hyunseokkkkkk.github.io/Humanoid_Footstep_Planning_GCS/>

The page uses static HTML, CSS, and JavaScript in `dist/`. It has no frontend
dependencies. Its source lives alongside the research code on `main`.

## Preview locally

From the repository root, run:

```bash
python3 website/build.py
python3 -m http.server 8765 --bind 127.0.0.1 --directory website/dist
```

Open <http://localhost:8765/>. The build stages the canonical manuscript and
supplementary video and checks local links, figure descriptions, and HTML IDs.
The two staged files are ignored to avoid storing duplicate large assets in Git.

## Publish with GitHub Pages

In the repository's **Settings → Pages → Build and deployment**, set **Source**
to **GitHub Actions** once. The workflow in
[paper-page.yml](../.github/workflows/paper-page.yml) publishes `website/dist`
whenever page sources or the canonical manuscript/video change on `main`.
It can also be run manually from the Actions tab. No extra secrets are needed.

If the first workflow ran before Pages was enabled, rerun it after changing the
setting. The `github-pages` environment records the deployed URL.

## Content and asset sources

- Manuscript: `paper/26-2531_03_MS.pdf`, copied as `assets/paper.pdf` at build time.
- Full video: `paper/Video_Final.mp4`, copied as `assets/overview.mp4` at build time.
- Figures: rendered from the original PDFs in `paper/source/Fig/` and saved as
  WebP images. Captions identify the published manuscript figures. They are
  published results, rather than claims that the repository regenerates every
  figure. See [the reproduction guide](../docs/reproduction.md) for that scope.
- The muted 16-second teaser comes from 3:16–3:32 of the supplementary video,
  cropped to its hardware footage (`1072:602:422:270`). Poster images also come
  from that video. The full video is unchanged.
- `assets/citation.bib` matches the manuscript's accepted publication status.
  Add DOI, volume, and pages here, in `index.html`, and in the root README when
  the final publication metadata becomes available.

The page is inspired by the academic project-page structure of
[DynaFlow](https://sowoolee.github.io/dynaflow/). Its implementation, figures,
footage, and text belong to this project; no DynaFlow code or media is bundled.
