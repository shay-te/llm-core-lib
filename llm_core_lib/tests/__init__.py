"""Test-package marker.

Installs stdlib stubs for the ``core_lib`` framework when it isn't on
the path. ``llm-core-lib``'s production code imports a handful of
``core_lib`` classes (the ``StatusCodeException`` base for typed
errors, the ``ConnectionFactory`` / ``Connection`` ABCs that the
``*ConnectionFactory`` / ``*Connection`` classes extend, and the
``CoreLib`` composition-root base for ``LlmCoreLib``). Real ``core_lib``
isn't installed in every test environment (notably the local coverage
workflow), so we register lightweight stubs at test-package import
time — when ``core_lib`` IS installed (the CI default), the ``try``
branch imports the real classes and the stubs never fire.

The stubs match the minimum surface our production code actually
uses; they are NOT a full reimplementation of the framework.
"""
from __future__ import annotations

try:  # pragma: no cover - exercised in CI where core_lib is installed
    from core_lib.error_handling.status_code_exception import (  # noqa: F401
        StatusCodeException,
    )
    from core_lib.connection.connection_factory import (  # noqa: F401
        ConnectionFactory,
    )
    from core_lib.connection.connection import Connection  # noqa: F401
    from core_lib.core_lib import CoreLib  # noqa: F401
except ImportError:
    import sys
    import types

    # ----- module skeleton -----
    _core_lib_module = types.ModuleType('core_lib')
    _error_handling_module = types.ModuleType('core_lib.error_handling')
    _status_code_exception_module = types.ModuleType(
        'core_lib.error_handling.status_code_exception'
    )
    _connection_pkg = types.ModuleType('core_lib.connection')
    _connection_factory_module = types.ModuleType(
        'core_lib.connection.connection_factory'
    )
    _connection_module = types.ModuleType('core_lib.connection.connection')
    _core_lib_main_module = types.ModuleType('core_lib.core_lib')

    # ----- stub classes -----

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

    class ConnectionFactory(object):  # type: ignore[no-redef]
        """Stub of :class:`core_lib.connection.connection_factory.ConnectionFactory`.

        The real ABC declares ``get()`` and ``close()``; our subclasses
        in ``llm_core_lib/connections/`` override both. We only need
        the class to exist as an inheritable base for ``isinstance``
        checks and MRO resolution.
        """

    class Connection(object):  # type: ignore[no-redef]
        """Stub of :class:`core_lib.connection.connection.Connection`.

        Same role as ``ConnectionFactory`` — an inheritable base. The
        real class implements ``__enter__`` / ``__exit__`` to support
        ``with factory.get() as connection`` flow; our concrete
        ``*Connection`` classes provide their own implementations.
        """

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            close = getattr(self, 'close', None)
            if callable(close):
                close()
            return False

    class CoreLib(object):  # type: ignore[no-redef]
        """Stub of :class:`core_lib.core_lib.CoreLib`.

        The real base wires the event-listener system and exposes
        ``connection_factory_registry`` etc.; our ``LlmCoreLib``
        composition root only calls ``CoreLib.__init__(self)`` and
        otherwise stays self-contained, so a no-op constructor is
        sufficient.
        """

        def __init__(self) -> None:
            pass

    # ----- wire stub modules together -----
    _status_code_exception_module.StatusCodeException = StatusCodeException
    _connection_factory_module.ConnectionFactory = ConnectionFactory
    _connection_module.Connection = Connection
    _core_lib_main_module.CoreLib = CoreLib

    _error_handling_module.status_code_exception = _status_code_exception_module
    _connection_pkg.connection_factory = _connection_factory_module
    _connection_pkg.connection = _connection_module

    _core_lib_module.error_handling = _error_handling_module
    _core_lib_module.connection = _connection_pkg
    _core_lib_module.core_lib = _core_lib_main_module

    sys.modules.setdefault('core_lib', _core_lib_module)
    sys.modules.setdefault('core_lib.error_handling', _error_handling_module)
    sys.modules.setdefault(
        'core_lib.error_handling.status_code_exception',
        _status_code_exception_module,
    )
    sys.modules.setdefault('core_lib.connection', _connection_pkg)
    sys.modules.setdefault(
        'core_lib.connection.connection_factory',
        _connection_factory_module,
    )
    sys.modules.setdefault('core_lib.connection.connection', _connection_module)
    sys.modules.setdefault('core_lib.core_lib', _core_lib_main_module)
