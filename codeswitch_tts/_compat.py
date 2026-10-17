"""Runtime compatibility helpers for the CodeSwitch-TTS package.

Kaggle's base image ships a protobuf 4.x runtime, while several preinstalled
packages (tensorflow 2.20, grpcio-tools, ydf, a2a-sdk, google-cloud-*) require
protobuf >= 5 and contain protobuf ``*_pb2.py`` "gencode" generated with
protoc >= 5.26. That gencode starts with:

    from google.protobuf import runtime_version as _runtime_version
    _runtime_version.ValidateProtobufRuntimeVersion(
        _runtime_version.Domain.PUBLIC, 5, 29, 5, "", "<file>_pb2.py"
    )

If the installed runtime is protobuf 4.x (no ``google.protobuf.runtime_version``
module) every one of those imports dies. The first version of this workaround
installed a fake module whose ``Domain`` was a *function*, which produced:

    AttributeError: 'function' object has no attribute 'PUBLIC'

The helper below installs an API-correct stand-in instead: ``Domain`` is a
class exposing ``PUBLIC`` / ``GOOGLE_INTERNAL`` exactly like the real
``google.protobuf.runtime_version.Domain`` enum, and version validation is a
no-op. This is a safety net for stale Kaggle sessions; the real fix is to keep
protobuf >= 5.26 installed (see CodeSwitch_TTS_Kaggle.ipynb, which pins
``protobuf==5.29.5`` and restarts the kernel afterwards).
"""

from __future__ import annotations

import os
import sys
import types


def install_protobuf_runtime_version_fallback() -> None:
    """Ensure ``google.protobuf.runtime_version`` is importable and API-correct.

    Does nothing when the installed protobuf runtime already ships
    ``runtime_version`` (protobuf >= 5.26). Otherwise registers a minimal
    compatible module in ``sys.modules`` so that modern protobuf gencode can be
    imported on top of an older runtime.
    """
    # Pure-Python protobuf implementation is the most tolerant one; only takes
    # effect when this line runs before google.protobuf is imported.
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

    try:
        import google.protobuf  # noqa: F401
    except Exception:
        # protobuf not installed at all - nothing to patch.
        return

    try:
        from google.protobuf import runtime_version  # noqa: F401
        return  # Real module present (protobuf >= 5.26): leave it alone.
    except ImportError:
        pass

    # --- Minimal API-compatible stand-in -------------------------------------
    class Domain:  # mirrors the real enum: runtime_version.Domain.PUBLIC
        GOOGLE_INTERNAL = 1
        PUBLIC = 2

    class Version:  # accepted by ValidateProtobufRuntimeVersion signatures
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    module = types.ModuleType("google.protobuf.runtime_version")
    module.Domain = Domain
    module.Version = Version
    module.DOMAIN = Domain.PUBLIC
    module.OSS_DOMAIN = Domain.PUBLIC
    module.OSS_MAJOR = 4
    module.OSS_MINOR = 25
    module.OSS_PATCH = 0
    module.OSS_SUFFIX = ""
    module.VersionError = Exception
    module.ValidateProtobufRuntimeVersion = lambda *args, **kwargs: None

    google.protobuf.runtime_version = module
    sys.modules["google.protobuf.runtime_version"] = module
