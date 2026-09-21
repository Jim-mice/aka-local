# Security policy

Never commit passwords, API keys, bearer tokens, SSH private keys, `.env` files, or real remote configuration. Local V100 configuration belongs in ignored `config/environments/v100.yaml`; `.venv/`, profiler binaries, and local artifacts are also ignored.

Use SSH keys or interactive authentication for remote access. If a secret is committed or published, revoke/rotate it immediately and report the incident privately to the repository maintainers. Removing it in a later commit is not sufficient because Git history may retain it.
