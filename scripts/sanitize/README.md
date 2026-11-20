# Sanitizer support files

`lsan.supp` is the LeakSanitizer suppression list applied by `make sanitize`
(through `NOVA_SANITIZE_LSAN_OPTIONS`). Keep it to third-party libraries: every
entry there is a finding the project cannot fix, and adding a project entry turns
a real leak into a green run. See `docs/ci.md`.
