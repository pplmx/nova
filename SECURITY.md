# Security Policy

## Supported Versions

The Nova project is a pre-release CUDA library (v0.1.0). All security vulnerabilities should be reported and treated with high priority.

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

Please report security vulnerabilities via the project's GitHub issue tracker:

https://github.com/pplmx/nova/issues

Please include the following in your report:

- A description of the vulnerability
- Steps to reproduce or proof-of-concept code
- Potential impact assessment
- Any suggested mitigations

## Response Process

- Reports are reviewed within 48 hours.
- If a vulnerability is confirmed, a patch will be developed in a private branch.
- A security advisory will be published on the GitHub repository.
- Users will be notified via the advisory system and encouraged to upgrade.

## Scope

This policy covers the Nova CUDA library source code in this repository. It does not cover:

- Vulnerabilities in third-party dependencies (e.g., CUDA runtime, cuBLAS)
- Issues in forks or modified versions of this code
