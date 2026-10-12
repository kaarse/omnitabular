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
npm run lint
npm run build
npm start
```

For static hosting, build with `NEXT_OUTPUT_EXPORT=1 npm run build` and serve
`out/`. Set `NEXT_PUBLIC_BASE_PATH=/omnitabular` when hosting at that subpath.
The GitHub Pages workflow sets these variables automatically. A deployed
website needs a reachable HTTPS API URL; localhost refers to the visitor's
computer.
