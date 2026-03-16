"""RoboMME package initialization."""

from __future__ import annotations


def _patch_maniskill_pci_render_backend() -> None:
    """Allow ManiSkill to pass PCI-style Vulkan device strings through intact."""
    try:
        from mani_skill.envs.utils.system import backend as ms_backend
    except Exception:
        return

    if getattr(ms_backend, "_robomme_pci_backend_patch", False):
        return

    original = ms_backend.parse_backend_device_id

    def patched_parse_backend_device_id(backend: str):
        if isinstance(backend, str) and backend.startswith("pci:"):
            return backend, None
        return original(backend)

    ms_backend.parse_backend_device_id = patched_parse_backend_device_id
    ms_backend._robomme_pci_backend_patch = True


_patch_maniskill_pci_render_backend()

