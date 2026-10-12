# Publish OmniTabular to PyPI

Installing with `python -m pip install omnitabular` requires an uploaded release on PyPI. A GitHub checkout or GitHub release alone does not make the package available there.

## One-time account setup

The workflow at `.github/workflows/publish.yml` supports either of these authentication methods:

- **Trusted Publishing:** In your PyPI account, add a pending publisher for a new project (or a publisher under the existing project's Publishing settings). Set the project name to `omnitabular`, owner to `kaarse`, repository to `omnitabular`, workflow filename to `publish.yml`, and environment to `pypi`. Create the matching `pypi` environment in GitHub repository settings.
- **API token:** Set the GitHub Actions secret `PYPI_API_TOKEN` to a PyPI token authorized to upload this project. The first upload of a new project requires an account-scoped token; after creation, replace it with a project-scoped token. The workflow uses this secret when present and otherwise uses Trusted Publishing.

See PyPI's [new project setup](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/) and [existing project setup](https://docs.pypi.org/trusted-publishers/adding-a-publisher/). Account setup requires the maintainer's PyPI access; it cannot be completed by editing this repository.

## Release

1. Set `__version__` in `omnitabular/__init__.py` to the new version. Build metadata reads that value without importing the ML dependencies.
2. Push the reviewed changes, then publish a GitHub release with a matching tag, such as `v0.2.0`. The release tag must point to the commit containing those changes. Alternatively, run **Publish to PyPI** manually from the Actions page, selecting the intended release ref.
3. The workflow builds the source archive and wheel, validates metadata, installs the wheel, and checks imports outside the checkout before uploading those artifacts to PyPI.
4. After publication, verify from a fresh virtual environment:

```bash
python -m venv /tmp/omnitabular-check
source /tmp/omnitabular-check/bin/activate
python -m pip install --index-url https://pypi.org/simple omnitabular==0.2.0
python -c "from omnitabular import OmniTabular; print(OmniTabular.available_models())"
```

PyPI does not allow replacing an uploaded distribution with the same filename. For subsequent releases, increment the version first. Once the first release is available, update the README and website installation commands to `python -m pip install omnitabular`.
