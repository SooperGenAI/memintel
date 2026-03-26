# Memintel Docs

Documentation site for [Memintel](https://github.com/SooperGenAI/memintel) — built with [Docusaurus](https://docusaurus.io) and [docusaurus-plugin-openapi-docs](https://github.com/PaloAltoNetworks/docusaurus-openapi-docs).

## Structure

```
static/api/
  openapi.yaml           ← OpenAPI 3.1 spec (App Developer API)

docs/
  intro/                 ← Introduction and core concepts
  api-reference/
    *.md                 ← Hand-written guides (tasks, execution, conditions…)
    generated/           ← Auto-generated from openapi.yaml — DO NOT EDIT
  python-sdk/            ← Python backend SDK reference

src/
  pages/                 ← Custom homepage
  css/                   ← Custom theme (IBM Plex Mono, Syne, indigo/cyan)

.github/workflows/
  deploy-docs.yml        ← Auto-deploy to GitHub Pages on push to main
```

## Local Development

```bash
npm install

# Generate interactive API reference from openapi.yaml
npm run gen-api-docs

# Start dev server
npm start               # → http://localhost:3000/memintel/
```

> **Important:** always run `gen-api-docs` before `start` or `build` after
> updating `static/api/openapi.yaml`. The generated files in
> `docs/api-reference/generated/` are committed to the repo so the site builds
> without the generation step in read-only environments.

## Updating the OpenAPI Spec

1. Edit `static/api/openapi.yaml`
2. Run `npm run gen-api-docs` — rewrites `docs/api-reference/generated/`
3. Commit both the YAML and the generated files
4. Push — CI deploys automatically

## Build

```bash
npm run gen-api-docs   # regenerate if spec changed
npm run build          # output in ./build
npm run serve          # preview production build locally
```

## Deployment

Docs deploy automatically to GitHub Pages on every push to `main` that touches
`docs/`, `src/`, `static/`, or config files. The CI pipeline runs
`gen-api-docs` before building.

**Enable GitHub Pages first:**
Settings → Pages → Source → **GitHub Actions**

Manual deploy:
```bash
GIT_USER=<your-github-username> npm run deploy
```

## Contributing Docs

- **Guides** — edit files in `docs/api-reference/*.md` or `docs/python-sdk/*.md`
- **API spec** — edit `static/api/openapi.yaml`, then run `gen-api-docs`
- **Generated files** — never edit `docs/api-reference/generated/` by hand
- **New pages** — add the file, then add it to the relevant sidebar in `sidebars.ts`
- Open a PR — CI validates the build before merge

