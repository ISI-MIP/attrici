# Development


## Documentation


Docstrings follow the [Numpy convention](https://numpydoc.readthedocs.io/en/latest/format.html).


## Tests

Run tests with

```
pytest tests
```

## Code formatting and linting

Attrici uses [ruff](https://docs.astral.sh/ruff/) for code formatting and linting.

To auto-format the code:

```
ruff format
```

To check code issues:

```
ruff check
```

Often style issues can automatically be fixed with

```
ruff check --fix
```

## Housekeeping

In some places versions need to be updated occasionally.

Python versions which should be tested against are defined in [.github/workflows/ci.yml](.github/workflows/ci.yml).
This might be useful to set to the versions used by (most) developers or users or those used on a cluster.
The supported versions are also defined in [pyproject.toml](pyproject.toml).
For the general support status of Python versions see the page of the [Python release cycle](https://devguide.python.org/versions/).

In [.github/workflows/ci.yml](.github/workflows/ci.yml) the GitHub actions version will require an occasional update to newer versions.
