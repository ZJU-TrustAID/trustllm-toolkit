# Security Policy

MarkSec is a research toolkit for evaluating the robustness of LLM watermarking
systems. Use it only on systems, models, and datasets where you have permission
to run evaluation workloads.

## Reporting a Vulnerability

If you find a vulnerability in MarkSec itself, please report it privately to the
maintainers before public disclosure. If no private security contact has been
published for the repository yet, open a GitHub security advisory or contact the
repository owner through the hosting platform.

Please include:

- Affected commit or release.
- Minimal reproduction steps.
- Impact and affected configuration.
- Whether model weights, datasets, API credentials, or local services are
  involved.

## Secrets and Local Artifacts

Never commit API keys, Hugging Face tokens, database passwords, `.env` files,
downloaded model weights, raw generated outputs, or local database files. The
default release configuration writes local artifacts under ignored paths such as
`output/`.
