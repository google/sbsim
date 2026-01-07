# Maintainer's Guide

This internal document provides instructions for Google employees to maintain
the open source "sbsim" codebase.

## Codebase Location

Navigating to the repo:

```sh
# read-only version:
cd /google/src/files/head/depot/google3/third_party/py/smart_buildings/

# editable version (using a CitC workspace):
cd /google/src/cloud/<username>/<client-name>/google3/third_party/py/smart_buildings/
# or:
cd third_party/py/smart_buildings/
```

## VS Code

We are encouraging open source contributors to use VS code as their default text
editor for working on this project.

For Google employees who would also like to use VS Code as their default editor,
see go/vscode/remote_development_via_web#serve-web for setup instructions.
This involves setting up a "~/.config/systemd/user/code.service" file
with the provided contents.

After setting up the "~/.config/systemd/user/code.service" file,
or after editing it, reload the service to have the changes take effect:

```sh
systemctl --user daemon-reload
```

Then you can use the following commands to manage the service:

```sh
# checking the status of the service:
systemctl --user status code

# stopping the service:
systemctl --user stop code

# starting the service:
systemctl --user start code

# restarting the service:
systemctl --user restart code
```

> NOTE: if you stop and start the service too quickly, it might cause an error,
> so it might help to wait a few seconds in between starting and stopping.

Once the service is running, to open VS Code, it might sometimes be possible
to visit the localhost address through the browser on your Chromebook, using a
port forwarding approach (see SSH connection below),
however this may not work reliably:

```sh
ssh <username>@<username>.c.googlers.com -L 59005:localhost:59005
```

In practice it may be more reliable to visit the localhost address
on your Cloudtop machine, through Chrome Remote Desktop.

Once you have accessed VS Code through the browser, it may be helpful to install
it as a progressive web app, so you can use it through its own dedicated
application window.

## Google-specific Style Checking

Google uses [`gpylint`](https://goto.google.com/gpylint) (a wrapper
around `pylint`) to check for `pylint`-related errors as well as additional
Google-specific code formatting errors that `pylint` does not handle. These
Google-specific errors begin with "g-" and can be ignored / disabled using the
usual `pylint` pragma comments (see README).

The `gpylint` checks are performed automatically by internal Google tools,
including during a Copybara sync (see "Copybara Sync" section below). Since
open source contributors are currently unable to run these checks, we need to
run them ourselves.

To run the style checker manually:

```sh
# check all files:
gpylint smart_control --ignore=proto

# check a specific file:
gpylint smart_control/path/to/file.py
```

This may produce verbose outputs, which may be helpful for specific errors but
which may be overwhelming when there are many errors. To control and reduce the
format of the error messages:

```sh
gpylint smart_control --ignore=proto --msg-template="{path}:{line}: [{msg_id}({symbol})]"
```

To ignore and/or check for certain messages, using the corresponding
[message name(s)](https://goto.google.com/gpylint-faq#rules):

```sh
# disabling certain messages (this is our default run command):
gpylint smart_control --ignore=proto --disable=g-bad-import-order,g-bad-todo --msg-template="{path}:{line}: [{msg_id}({symbol})]"

# checking for a specific message:
gpylint smart_control --ignore=proto --disable=all --enable=g-doc-args
```

The `gpylint` tool doesn't use the existing ".pylintrc"
config file, so we have [created](https://critique.corp.google.com/cl/758181505) our own [custom `gpylint` configuration](https://source.corp.google.com/piper///depot/google3/devtools/gpylint/config/oss_smart_buildings/), which ignores certain
messages such as "g-bad-import-order", which allows us to group local imports
in their own section at the bottom.

To apply our custom `gpylint` config, run it in "oss_smart_buildings" mode. This
is the mode that gets run on CL pre-submit:

```sh
gpylint smart_control --ignore=proto --mode=oss_smart_buildings
```

## Copybara Sync

We are using [Copybara](https://github.com/google/copybara) to manage the code
sync process between GitHub (open source) and Google (internal) codebases.

### Setup Copybara

Follow the [Copybara Setup](https://goto.google.com/copybara-setup) instructions
to setup Copybara on your Google machine.

This involves setting a bash alias for the `copybara` CLI.

If successful, these commands should resolve without error:

```sh
copybara version

copybara help
```

### Configure Copybara

Ensure the "copy.bara.sky" file in the root directory of the repository is up to
date. This file exists in the Google codebase only.

Navigate to the repository from the command-line before running any of the
Copybara commands below that reference the "copy.bara.sky" config file.

See more information about the existing workflows:

```sh
copybara info copy.bara.sky
```

You will need to use your judgement about which workflow needs to be run,
and the order in which to run them.

### GitHub to Google

Run the "default" workflow to perform a dry run sync from GitHub to Google:

```sh
# sync from the default branch:
copybara copy.bara.sky  --init-history --dry-run

# sync from a specific branch:
copybara copy.bara.sky --init-history --dry-run default <branch-name>

# sync from a specific commit:
copybara copy.bara.sky --init-history --dry-run default <commit-SHA>
```

This creates a CL on the Google side, pulling in all the changes from GitHub to
Google.

> NOTE: the GitHub to Google sync process updates the "METADATA" file
> automatically, flipping the "Piper" block's `primary_source` setting to
> be `false`, and adding a "Git" block with a `primary_source` setting of
> `true`.

> NOTE: the "Git" block includes information related to the latest release,
> so it may be helpful to create a new release tag on GitHub before
> performing a sync. Although, a sync was performed before we made any
> releases, and we have anecdotal evidence that a release was being referenced
> even after it was deleted on GitHub, so additional investigation into the
> relationship between Copybara and the release tags may be helpful.

> NOTE: any changes made to the "METADATA" file from GitHub will be
> overridden, so if you need to make updates to that file, you will need to
> update it on the Google side.

### Google to GitHub

#### Setup GitHub Credentials

When pushing code to GitHub, it will ask you to provide your username and
password, or use a personal access token.

Obtaining a Personal Access Token:

  + Go to your GitHub settings, then "Developer settings," and then "Personal access tokens (classic)".
  + Click "Generate new token".
  + Give the token a descriptive name (e.g. "Copybara Sync").
  + Grant the `public_repo` permission under the repo scope.
  + Copy the generated token.

Create or update the "~/.googlekeys/copybara_git_credentials" file on your
Cloudtop to include your token in the following format:

```
https://<YOUR_GITHUB_USERNAME>:<YOUR_TOKEN>@github.com
```

#### Running the Workflow

Once your credentials are in place, you can run the "piper_to_github_pr"
workflow:

```sh
copybara copy.bara.sky piper_to_github_pr --ignore-noop

# if the last change was to a file not included in GitHub:
copybara copy.bara.sky piper_to_github_pr --ignore-noop --last-rev <CL_NUMBER_BEFORE_YOUR_COPY.BARA.SKY_CHANGE>

# to use the last n=8 Google commits:
copybara copy.bara.sky piper_to_github_pr --ignore-noop --iterative-limit-changes 8
```

This will create a Pull Request the GitHub repository.
