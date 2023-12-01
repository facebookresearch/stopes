# Contributing to `stopes`

We want to make contributing to this project as easy and transparent as
possible.

## Our Development Process

`stopes` is built as part of the FAIR NLLB project. We work on `stopes` internally
and push our changes to the open source community when we have a stable version
or interesting results.

## Pull Requests

We actively welcome your pull requests.

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. If you haven't already, complete the Contributor License Agreement ("CLA").

## Contributor License Agreement ("CLA")

In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Meta's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## Issues

We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

Meta has a [bounty program](https://www.facebook.com/whitehat/) for the safe
disclosure of security bugs. In those cases, please go through the process
outlined on that page and do not file a public issue.

## License

By contributing to `stopes`, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.

## Code style

To control code style, we use several automatic tools:

- [black](https://black.readthedocs.io/en/stable/) for general styling
- [isort](https://pycqa.github.io/isort/) for sorting import statements
- [mypy](https://mypy.readthedocs.io/en/stable/) for static type checking

`black` and `isort` are able to modify your files to comply with their standards,
whereas `mypy` only tells about potential inconsistencies in the types.

We suggest our contributors to form a habit of using these tools.
Before merging each commit, we are checking the code with them automatically,
so they are mandatory for contributing any changes. Some of them are integrated into pre-commit hooks (see below), so you can setup their automatic execution for each commit.

Additionally, we are checking that all Python files have correct copyright headers.
To automatically fix these headers, you can run

```
python stopes/core/tests/test_headers.py
```

## Pre-commit hooks

In order to ensure your code lints, there are pre-commit hooks configured in the repository which you can install.
After installation, they will automatically run each time you commit.
An abbreviated guide is given below; for more information, refer to [the offical pre-commit documentation](https://pre-commit.com/).

### Installation

```
pip install pre-commit
pre-commit install
```

### Usage

Just commit your changes using command line (or your IDE, if you prefer):

```
git commit -m "My informative commit message"
```

If there was a failure, you will get feedback.

Certain hooks modify your files to comply.
To include these modifications, you will need to add them (i.e. `git add ...`) and commit again.

## Testing

The tests in `stopes` can be run with `pytest`.
It can be installed with

```
pip install pytest
```

To run the full test suite (which takes a few minutes), just type

```
pytest
```

To run a single particular test (e.g. after fixing it), you can run a command like

```
pytest stopes/pipelines/tests/test_configs.py::test_configs
```

For more details, please read the [pytest official documentation](https://docs.pytest.org/).

If you are an internal contributor at FAIR, please also consider running the integration tests.
