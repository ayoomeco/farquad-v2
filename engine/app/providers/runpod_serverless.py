"""RunPod Serverless (Flex workers + FlashBoot + network volume). DRY-RUN planner in this build: builds the
endpoint/template spec and per-job cost; never calls the API. Phase 3 target for production (scale-to-zero)."""
from __future__ import annotations
from .base import Provider, ProviderError


class RunPodServerlessProvider(Provider):
    name = 'runpod_serverless'

    def plan_endpoint(self, image: str, network_volume_id: str | None, gpu_ids: list[str] | None = None) -> dict:
        c = self.cfg
        return {'status': 'NOT CREATED (dry-run)',
                'template': {'name': 'jutsu-worker', 'imageName': image, 'isServerless': True, 'containerDiskInGb': 20,
                             'env': {'COMFY_PORT': '8188'}},
                'endpoint': {'name': 'genjutsu', 'gpuTypeIds': gpu_ids or [g['id'] for g in self.gpus()],
                             'networkVolumeId': network_volume_id, 'workersMin': 0, 'workersMax': int(c.get('workers_max', 1)),
                             'idleTimeout': int(c.get('idle_timeout_s', 5)), 'flashboot': True,
                             'executionTimeoutMs': int(c.get('execution_timeout_s', 1800)) * 1000, 'scalerType': 'QUEUE_DELAY'},
                'notes': ['Flex workers bill per second only while running; FlashBoot gives fast warm starts',
                          'models come from the network volume mounted at /runpod-volume (worker/start.sh finds it)']}

    def launch(self, *a, **k):
        raise ProviderError('runpod_serverless live dispatch is not implemented in this build (dry-run planner only)')
