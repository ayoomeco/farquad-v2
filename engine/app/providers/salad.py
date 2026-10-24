"""Salad Cloud container group (RTX 5090). DRY-RUN planner: kept for cheap batch work ($0.25-0.50/hr) once
the worker image is published. Never calls the Salad API in this build."""
from __future__ import annotations
from .base import Provider, ProviderError


class SaladProvider(Provider):
    name = 'salad'

    def plan_group(self, image: str, gpu: str = 'rtx5090') -> dict:
        c = self.cfg
        return {'status': 'NOT CREATED (dry-run)', 'org': c.get('org'), 'project': c.get('project'),
                'container_group': {'name': 'jutsu-worker', 'image': image, 'replicas': 1, 'gpu_classes': [gpu],
                                    'priority': c.get('priority', 'high'), 'ports': [8188], 'restart_policy': 'never'},
                'notes': ['Salad has no network volumes: models download per node at boot (or bake a models image)',
                          'per-second billing while running; stop the group when the job ends']}

    def launch(self, *a, **k):
        raise ProviderError('salad live dispatch is not implemented in this build (dry-run planner only)')
