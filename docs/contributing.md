# Contributor's Guide

We welcome your contributions to this repository!

Consult the sections below for more information about the contribution process,
including guidelines about code documentation, testing, and formatting.

## Contributor License Agreement

Contributions to this project must be accompanied by a
[Contributor License Agreement](https://cla.developers.google.com/about) (CLA).
You (or your employer) retain the copyright to your contribution; this simply
gives us permission to use and redistribute your contributions as part of the
project.

If you or your current employer have already signed the Google CLA (even if it
was for a different project), you probably don't need to do it again.

Visit <https://cla.developers.google.com/> to see your current agreements or to
sign a new one.

## Community Guidelines

This project follows
[Google's Open Source Community Guidelines](https://opensource.google/conduct/).

See also the [Code of Conduct](./code-of-conduct.md).

## Code Reviews and Pull Request Workflow

All submissions, including submissions by project members, require review. We
use [GitHub pull requests](https://docs.github.com/articles/about-pull-requests)
for this purpose.

## Documentation

We encourage you to document your code using docstrings and type hints.
Specifically we use the
[Google Docstring Guidelines](https://google.github.io/styleguide/pyguide.html#381-docstrings)
outlined in the Google Python Style Guide.

Here are some additional
[examples of Google-formatted docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).

The content of the [documentation site](./docs-site.md) is automatically
generated based on these docstrings.

## Testing

We encourage you to add tests to ensure your code is working as expected.

Tests should be placed in a new file next to the file under test. The test file
name should be the same as the name of the file under test, with "\_test.py"
appended at the end. For example, if you are testing a file called
"my_library.py", the test file should be named "my_library_test.py".

We primarily leverage the `unittest`, `absltest`, and `tf.test` frameworks for
writing tests, and we use the `pytest` tool for running tests.

See existing test files for example structure. Here is a simplified example:

```py
# this is an example "_test.py" file...

from absl.testing import absltest

class CalculatorTest(absltest.TestCase):

  def test_addition(self):
    self.assertEqual(2+2, 4)

if __name__ == "__main__":
  absltest.main()
```

Running tests:

```sh
# run all tests:
pytest

# disable warnings:
pytest --disable-pytest-warnings

# show print statements:
pytest --disable-pytest-warnings -s

# run specific test files:
pytest --disable-pytest-warnings path/to/your/test.py

# run specific test class:
pytest --disable-pytest-warnings path/to/your/test.py::YourUnittestClass

# run specific tests:
pytest --disable-pytest-warnings -k your_test_name_here

# ignore specific test files and directories:
pytest --ignore=path/to/your/test.py --ignore=path/to/other/

```

## Linting

### Python Style Formatting

We are using [`pyink`](https://github.com/google/pyink) to format Python code
according to
[Google Python Style Guidelines](https://google.github.io/styleguide/pyguide.html).
The formatter will automatically update files inplace.

The formatter will run automatically as a pre-commit hook (see "Pre-commit
Hooks" section below for more information and setup instructions).

Additionally, for contributors using the VS Code text editor, we have configured
a VS Code workspace settings file to run the formatter whenever a file is saved.
NOTE: this requires the
[`ms-python.black-formatter` extension](https://marketplace.visualstudio.com/items?itemName=ms-python.black-formatter)
for VS Code.

If you would like to run the formatter manually:

```sh
# format all the files:
pyink .

# format a specific file or directory:
pyink /path/to/file/or/dir
```

If you would like to perform a dry run:

```sh
# check if a file would be changed:
pyink . --check

# see what changes would be made:
pyink . --diff
```

If you would like to prevent certain lines of code from being formatted (for
example to leave a long line as-is), it is possible to
[ignore formatting](https://black.readthedocs.io/en/stable/usage_and_configuration/the_basics.html#ignoring-sections)
by adding a trailing comment of `# fmt: skip` to the right of the line / at the
end of the expression, or by wrapping multiple lines of code between
`# fmt: off` and `# fmt: on` comments. NOTE: `pyink` and `pylint` (see "Python
Style Checking" section below) may each require their own separate set of
comments, however `pyink` respects many `pylint` comments, so you are
recommended to try using a `pylint` comment first, and then only also add a
`pyink` comment as necessary.

### Python Import Sorting

We are using [`isort`](https://pycqa.github.io/isort/) to control the sort order
of Python import statements, specifically grouping the "smart_control" local
module imports separately in their own section below the package imports.

The import sorter will run automatically as a pre-commit hook (see "Pre-commit
Hooks" section below for more information and setup instructions).

Additionally, for contributors using the VS Code text editor, we have configured
a VS Code workspace settings file to run the import sorter whenever a file is
saved. NOTE: this requires the
[`ms-python.isort` extension](https://marketplace.visualstudio.com/items?itemName=ms-python.isort)
for VS Code.

If you would like to run the import sorter manually:

```sh
# sort all the files:
isort .

# sort a specific file:
isort /path/to/file.py

# sort with verbose outputs (helpful for troubleshooting):
isort -v .
```

### Python Style Checking

We are using [`pylint`](https://pylint.readthedocs.io/en/stable/index.html) to
check for additional Python style formatting issues that `pyink` doesn't fix, to
more closely follow
[Google Python style guidelines](https://google.github.io/styleguide/pyguide.html).
The style checker will NOT automatically update files inplace, but rather will
produce a report containing any errors that you will need to fix manually.

The style checker will run automatically as a pre-commit hook (see "Pre-commit
Hooks" section below for more information and setup instructions).

If you would like to run the style checker manually:

```sh
# check all files:
pylint --rcfile=.pylintrc --ignore=proto smart_control

# check a specific file:
pylint --rcfile=.pylintrc --ignore=proto smart_control/path/to/file.py
```

To check for a specific issue (e.g. "missing-module-docstring"), using the
corresponding
[message code](https://pylint.readthedocs.io/en/stable/user_guide/messages/messages_overview.html)
(e.g. "C0114"):

```sh
pylint smart_control --rcfile=.pylintrc --ignore=proto --disable=all --enable=C0114
```

If you would like to prevent certain lines of code from being checked (for
example to leave a long line as-is), it is possible to
[ignore formatting](https://pylint.readthedocs.io/en/stable/user_guide/messages/message_control.html#block-disables)
for a given message (e.g. "line-too-long") by adding a trailing comment of
`# pylint: disable=line-too-long` to the right of the line / at the end of the
expression, or by wrapping multiple lines of code between
`# pylint: disable=line-too-long` and `# pylint: enable=line-too-long` comments.

### Markdown Formatting

We are using [`mdformat`](https://github.com/hukkin/mdformat) to check for
formatting errors in markdown files.

The markdown formatter will run automatically as a pre-commit hook (see
"Pre-commit Hooks" section below for more information and setup instructions).

If you would like to run the markdown formatter manually:

```sh
# format specific file(s):
mdformat README.md docs/*.md

# check if a file would be changed:
mdformat README.md --check
```

> NOTE: we are ignoring markdown files in the "docs/api" directory because they
> contain [auto-documentation](./docs-site.md) formatting directives like `:::`
> that get improperly formatted if those directives contain additional
> configuration options.

> NOTE: it would be nice to check all markdown files, however this currently
> includes all files in the ".venv" folder (not desired), and the functionality
> for ignoring certain directories is only supported in Python 3.13+. When we
> upgrade, we can consider updating the approach, but right now we are only
> targeting specific files.

The `mdformat` tool might not be able to format certain long lines containing
code fences, so some manual review may still be required. Long lines caused by
links are OK to keep as-is.

## Pre-commit Hooks

We are using pre-commit hooks to perform code formatting, import sorting, and
style checking. These actions will take place on each commit.

To enable the pre-commit hooks, you must perform a one-time setup by running
`pre-commit install`. This will create or update ".git/hooks/pre-commit".

If you would like to run the pre-commit hooks without making a commit:

```sh
# run against staged files only:
pre-commit run

# run against all files:
pre-commit run --all-files

# run against a specific set of file(s):
pre-commit run --files path/to/my_file.py path/to/other_file.py
```

If you encounter issues and need to clear the cache:

```sh
pre-commit clean
```

If you would like to make a commit and skip the hooks (not recommended), use the
`--no-verify` flag with your `git commit` command.
