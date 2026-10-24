from __future__ import annotations
import time
from dataclasses import dataclass, field


class ProviderError(Exception):
    pass


class ApprovalError(ProviderError):
    pass


class BudgetExceeded(ProviderError):
    pass


@dataclass
class Approval:
    """the owner's explicit OK for ONE spend, with its own ceiling. Built only by an interactive confirm
    (bake-off runner) or the app's approve endpoint; never inferred."""
    job: str
    max_usd: float
    approved_by: str = 'the owner'
    confirmed: bool = False
    created: float = field(default_factory=time.time)


class Provider:
    name = 'base'
    live_capable = False

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def gpus(self) -> list[dict]:
        return list(self.cfg.get('gpus') or [])

    def price(self, gpu_id: str | None = None) -> float:
        g = self.gpus()
        for x in g:
            if gpu_id in (None, x.get('id'), x.get('name')):
                return float(x['usd_per_hr'])
        raise ProviderError(f'no price for gpu {gpu_id!r}')

    def quote(self, gpu_seconds: float, gpu_id: str | None = None, fixed_usd: float = 0.0) -> dict:
        rate = self.price(gpu_id)
        usd = gpu_seconds / 3600 * rate + fixed_usd
        return {'provider': self.name, 'gpu': gpu_id or self.gpus()[0].get('id'), 'usd_per_hr': rate,
                'gpu_seconds': int(gpu_seconds), 'usd': round(usd, 3), 'basis': self.cfg.get('price_basis', '')}

    def describe(self) -> dict:
        return {'name': self.name, 'mode': self.cfg.get('mode', 'dry_run'), 'live_capable': self.live_capable,
                'gpus': self.gpus(), 'hard_cap_usd': self.cfg.get('hard_cap_usd')}
