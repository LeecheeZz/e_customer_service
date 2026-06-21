"""Project entrypoint that delegates to the package CLI.

This file preserves the original `sft.py` location so existing workflows
can call `python sft.py ...` while the implementation is moved to
the `e_customer_service` package.
"""

from e_customer_service.cli import main


if __name__ == "__main__":
    main()
