# Security

## Supported versions

Only the `main` line (`0.9.1` workspace version) receives fixes.
Older Machi ≤0.8 is unsupported.
Historical crates.io `machi*` 1.0.0 and the `ovo` 0.9.0 stub are unsupported.

## Reporting

Report vulnerabilities privately to the maintainers via the repository’s
security advisory channel (GitHub Security Advisories) when available, or by
contacting the org listed in the repository metadata. Do not open public
issues for unfixed critical flaws.

## Dependency policy

```bash
cargo deny check
```

- **Advisories:** fail on known RustSec issues (see `deny.toml`).
- **Licenses:** explicit allow-list (MIT/Apache-2.0 family + common
  permissive deps used by TLS/ICU).
- **Bans / sources:** default deny-template rules; crates.io is the expected
  source for third-party crates.

## Runtime trust model

- Default path is **offline** (`MockSampler`); network providers are feature-gated.
- Toolkit tools are **cwd-jailed**; do not treat them as a full OS sandbox.
- Default isolation is **in-process** (`InProcessIsolation`). Product sandboxes
  inject a custom `IsolationBackend`.
- Budget, depth, concurrency, and journal divergence are **fail-closed**.
- Workflow engine never loads LLM HTTP clients (`ovo-workflow` firewall).
