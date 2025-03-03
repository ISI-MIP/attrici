# Development

## Install instructions for the development version

Clone the repository and checkout the current development branch.

Create a virtual environment, since the different optimization libraries don't work with all Python version it is
recommended to set the version explicitly.

```bash
python3.12 -m venv env
```

Activate the environment (adjust if using a non-default shell)

```bash
source env/bin/activate
```

Install ATTRICI as a local development version with development dependencies for testing, coverage, and linting included.

```bash
pip install -e .[dev]
```

To install the PyMC3 (which requires Python 3.11) and PyMC5 versions concurrently run

```bash
python3.11 -m venv env311
source env311/bin/activate
pip install -e .[dev,pymc3]
```

Note that the command line tools can be used in one environment with the different solvers but there might occur
problems when importing both PyMC3 and PyMC5 modules in one Python process.

Additional dependencies for Jupyter notebooks can be installed with

```bash
pip install -e .[dev,pymc3,notebook]
```

## Documentation

Documentation is generated with [pdoc](https://pdoc.dev/) for the latest commit on the `main` branch.
The readme, usage and development guides are imported from their Markdown files, see `attrici/__init__.py`.

These are followed by auto-generated API docs for the ATTRICI modules.

Docstrings follow the [Numpy convention](https://numpydoc.readthedocs.io/en/latest/format.html).
To check that all modules and functions are documented
[docstr-coverage](https://github.com/HunterMcGushion/docstr_coverage) can be used:

```bash
docstr-coverage attrici
```

Pdoc handles inherited docstrings. For example docstrings to be skipped for derived classed can be marked
for `docstr-coverage` with a comment before the function: `# docstr-coverage:inherited`.

Run the following command to generate the docs locally.

```bash
PDOC_ALLOW_EXEC=1 pdoc attrici --logo "https://avatars.githubusercontent.com/u/7933269?s=200&v=4" --docformat numpy --math
```

The `PDOC_ALLOW_EXEC` flag is required for importing and analyzing PyMC3 and its dependencies.
Note that the optional PyMC3 dependency works only with Python 3.11 and its dependencies
need to be installed separately.

## Tests

Run tests with

```bash
pytest tests
```

Through the [pytest-cov](https://github.com/pytest-dev/pytest-cov) plugin [coverage](https://coverage.readthedocs.io)
statistics are generated when running the tests with

```bash
pytest --cov=attrici
```

They can be viewed with

```bash
coverage report
```

or

```bash
coverage html
```

For files ignored from coverage checks, see the `tool.coverage.run` section in `pyproject.toml` for settings.
There is no enforced coverage rate during CI but using `coverage` can be helpful to find areas to improve tests for.

## Code formatting and linting

ATTRICI uses [ruff](https://docs.astral.sh/ruff/) for code formatting and linting.

To auto-format the code:

```bash
ruff format
```

To check code issues:

```bash
ruff check
```

Often style issues can automatically be fixed with

```bash
ruff check --fix
```

Occasionally, there might be an update of `ruff` with changes that might require to locally update the `ruff` version
to match the version used in CI on GitHub.

## Markdown linting

For keeping the Markdown documentation files in a consistent way
[PyMarkdown Linter](https://pypi.org/project/pymarkdownlnt/)
is used.
Run the following command to check the Markdown rules which can be adjusted in the `pyproject.toml` file.

```bash
pymarkdown scan *.md
```

See the [PyMarkdown Linter Documentation](https://pymarkdown.readthedocs.io/en/latest/rules/) for specific settings.
Note that the package is called pymarkdownlnt for historical reasons, as explained in its
[documentation](https://pymarkdown.readthedocs.io/en/latest/#why-is-this-application-referred-to-as-pymarkdown-and-pymarkdownlnt).
To suppress individual warnings, for example line length it is possible to
[set pragmas](https://pymarkdown.readthedocs.io/en/latest/advanced_plugins/#suppressing-rule-failures)

## Regularly needed updates

In some places in the code base version numbers need to be updated occasionally.

Python versions which should be tested against are defined in [.github/workflows/ci.yml](.github/workflows/ci.yml).
This might be useful to set to the versions used by (most) developers and users or those used on a cluster.
The supported versions are also defined in [pyproject.toml](pyproject.toml).
For the general support status of Python versions see the page of the
[Python release cycle](https://devguide.python.org/versions/).
Note that over time and with fading upstream support there might be incompatibilites
and unresolvable dependencies.
Some work-arounds and patches are already defined in `attrici.estimation.model_pymc3`
to allow for installation with newer versions of Python and Numpy.

In [.github/workflows/ci.yml](.github/workflows/ci.yml) the GitHub actions version will require an occasional
update to newer versions.
