"""Test-package marker.

Installs a ``core_lib.error_handling.status_code_exception`` stub when
``core_lib`` isn't on the path. ``llm-core-lib``'s
``safety/payload_gate.py`` imports ``StatusCodeException`` from
``core_lib`` so ``UnsafeToolResultError`` can inherit from it and pick
up the workspace-wide ``@HandleException`` HTTP mapping — but the
sibling-package isn't installed in every test environment (notably the
adversarial / coverage workflow that doesn't need the full framework).
The stub keeps the test suite self-contained without changing
production behavior: when ``core_lib`` IS installed (the CI default),
the ``try`` branch resolves the real class and the stub never fires.
"""
from __future__ import annotations

try:  # pragma: no cover - exercised in CI where core_lib is installed
    from core_lib.error_handling.status_code_exception import (  # noqa: F401
        StatusCodeException,
    )
except ImportError:
    import sys
    import types

    _core_lib_module = types.ModuleType('core_lib')
    _error_handling_module = types.ModuleType('core_lib.error_handling')
    _status_code_exception_module = types.ModuleType(
        'core_lib.error_handling.status_code_exception'
    )

    class StatusCodeException(Exception):  # type: ignore[no-redef]
        """Stub of :class:`core_lib.error_handling.status_code_exception.StatusCodeException`.

        Matches the signature the real framework uses
        (``StatusCodeException(status_code, message='')``) so subclasses
        like :class:`UnsafeToolResultError` construct identically. The
        stub IS an ``Exception`` so ``except Exception`` paths catch it,
        which is what the gate's sanitized-error flow relies on.
        """

        def __init__(self, status_code, message: str = '') -> None:
            super().__init__(message)
            self.status_code = status_code

    _status_code_exception_module.StatusCodeException = StatusCodeException
    _error_handling_module.status_code_exception = _status_code_exception_module
    _core_lib_module.error_handling = _error_handling_module

    sys.modules.setdefault('core_lib', _core_lib_module)
    sys.modules.setdefault('core_lib.error_handling', _error_handling_module)
    sys.modules.setdefault(
        'core_lib.error_handling.status_code_exception',
        _status_code_exception_module,
    )
