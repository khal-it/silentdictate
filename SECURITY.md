# Security policy

SilentDictate handles microphone audio and injects keystrokes, so security reports are taken seriously.

## Reporting a vulnerability

Please report privately through [GitHub's private vulnerability reporting](https://github.com/khal-it/silentdictate/security/advisories/new), not in a public issue.

This is a volunteer project, so response times aren't guaranteed. The latest release is the only supported version.

## What counts

- Audio or text leaving the machine without the user opting in
- Audio or dictated text written to disk or logs unexpectedly
- Enter being sent without `auto_submit`
- Other local users controlling the socket, or injection through crafted text
- Models or downloads that skip checksum verification

Bad transcriptions are bugs, not vulnerabilities.
