"""Vast.ai marketplace (RTX 5090 ~$0.40-0.53/h, cheapest burst source per the Oct 7 final sweep).
DRY-RUN planner only: builds the instance search filter + create spec, never calls the API. Key (VAST_API_KEY)
would come from the environment only. Caveats: per-host reliability/bandwidth vary, no shared network volumes
(models download per instance, so prefer hosts with >= 1 Gbps), and instances must be destroyed (not just stopped)
to stop storage billing."""
from __future__ import annotations
from .base import Provider, ProviderError


class VastProvider(Provider):
    name = 'vast'

    def search_filter(self, gpu: str = 'RTX 5090', disk_gb: int = 220) -> dict:
        c = self.cfg
        return {'gpu_name': {'eq': gpu}, 'num_gpus': {'eq': 1}, 'rentable': {'eq': True}, 'verified': {'eq': True},
                'reliability': {'gte': float(c.get('min_reliability', 0.98))},
                'inet_down': {'gte': int(c.get('min_inet_down_mbps', 1000))}, 'disk_space': {'gte': int(disk_gb)},
                'cuda_max_good': {'gte': 13.0}, 'order': [['dph_total', 'asc']], 'type': 'on-demand'}

    def plan_instance(self, image: str, disk_gb: int = 220, max_seconds: int | None = None) -> dict:
        env = {'MODEL_ROOT': '/workspace'}
        if max_seconds:
            env['JUTSU_MAX_SECONDS'] = str(int(max_seconds))
        return {'status': 'NOT CREATED (dry-run)', 'search': self.search_filter(disk_gb=disk_gb),
                'create': {'image': image, 'disk': int(disk_gb), 'env': env, 'onstart': 'bash /opt/jutsu-src/start.sh',
                           'runtype': 'ssh_direct', 'ports': [8188]},
                'notes': ['first boot runs worker/bootstrap_volume.py --group phase1 into /workspace (no network volumes on Vast)',
                          'destroy the instance afterwards: stopped instances still bill storage']}

    def launch(self, *a, **k):
        raise ProviderError('vast live dispatch is not implemented in this build (dry-run planner only)')
