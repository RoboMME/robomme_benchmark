# Mechanism implementation

See the [benchmark overview](../../README.md#task-changes) for the variation summary.
Episode metadata selects seeded motion, interaction or activation changes.

Overrides reuse task code in isolated module namespaces so they cannot modify
other episodes. Marked episodes use updated expert execution lengths; disabling
mechanism variations restores the original step budget. Recording tools and
expert execution policies stay outside this package.
