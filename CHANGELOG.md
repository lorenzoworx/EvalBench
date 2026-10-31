# Changelog

## Unreleased

- Run the static replay container as an unprivileged, read-only-compatible user.
- Deploy verified `main` commits through a dedicated production runner.
- Verify the replay fixture and write rejection before and after promotion, with
  automatic image rollback on failure.
