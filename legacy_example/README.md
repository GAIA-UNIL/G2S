# Legacy Examples

These examples preserve the pre-schema positional output style. Each script passes `-legacy_output` directly in its `g2s` calls, so unpacking patterns such as `simulation, index, *_ = g2s(...)` and `[simulation, index] = g2s(...)` continue to behave as before.

New examples live under `example/` and use schema output objects by default.
