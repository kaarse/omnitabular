# OmniTabular website

Requires Node.js 20.9+ and the Python API running on port 8080.
See the repository README for API setup.

```bash
npm ci
npm run dev
```

Open http://localhost:3000/dashboard. Set `NEXT_PUBLIC_API_URL` before startup
or build to use another API address. Configure `OMNITABULAR_CORS_ORIGINS` on
that API to allow the website origin.

```bash
npm test
npm run lint
npm run build
npm start
```

For static hosting, build with `NEXT_OUTPUT_EXPORT=1 npm run build` and serve
`out/`. Set `NEXT_PUBLIC_BASE_PATH=/omnitabular` when hosting at that subpath.
The GitHub Pages workflow sets these variables automatically. A deployed
website needs a reachable HTTPS API URL; localhost refers to the visitor's
computer.

Static exports use trailing slashes and generate a directory `index.html` for
each route so GitHub Pages can serve direct links and refreshes, including
`/omnitabular/dashboard/` and `/omnitabular/api-reference/`.


## Documentation updates

The page at `/api-reference` contains the setup guide, Python options, model
registry, REST API, deployment instructions, and troubleshooting. Its API data
is generated from Python source without installing ML dependencies:

```bash
# From the repository root
python scripts/build_docs.py
python scripts/build_docs.py --check
```

The generator updates `src/data/api-reference.json`; it preserves the page
layout and guide text. CI checks this data for drift and runs the CSV parser
tests, lint, and production build. No external font download is required.
