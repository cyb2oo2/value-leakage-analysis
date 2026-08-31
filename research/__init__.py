"""Read-only research tooling layered on top of the shipped experiment data.

Modules in this package must never mutate ``runs/``.  Derived artifacts belong
in ``derived/``, ``figures/``, or an explicitly supplied output directory.
"""

