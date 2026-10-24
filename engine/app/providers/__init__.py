"""GPU providers: runpod_pod (live-capable, approval-gated, watchdog + hard $ cap), runpod_serverless and
salad / vast (dry-run planners), local_omen (plan-only: the owner's RTX 2060). Keys are read from the environment at call time only."""
from __future__ import annotations
from .. import config


def provider_cfg(name: str) -> dict:
    return dict((config.load().get('providers') or {}).get(name) or {})


def get(name: str):
    if name == 'runpod_pod':
        from .runpod_pod import RunPodPodProvider
        return RunPodPodProvider(provider_cfg(name))
    if name == 'runpod_serverless':
        from .runpod_serverless import RunPodServerlessProvider
        return RunPodServerlessProvider(provider_cfg(name))
    if name == 'salad':
        from .salad import SaladProvider
        return SaladProvider(provider_cfg(name))
    if name == 'vast':
        from .vast import VastProvider
        return VastProvider(provider_cfg(name))
    if name == 'local_omen':
        from .local_omen import LocalOmenProvider
        return LocalOmenProvider(provider_cfg(name))
    raise KeyError(f'unknown provider {name!r}')


def all_names() -> list[str]:
    return sorted((config.load().get('providers') or {}).keys())
