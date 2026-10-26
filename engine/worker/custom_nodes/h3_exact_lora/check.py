"""Pure load-check for H3ExactLoraStack. No torch, no ComfyUI.

A render passes only when every requested gate holds: lossy merges at or under the cap
(a missing count is a failure), no load warnings, expected adapter counts, hook floors,
each named LoRA present at its set strength, the exact-stack injection present, and
ComfyUI at or above the minimum version.
"""


def version_tuple(value):
    parts = []
    for piece in str(value or '').split('.'):
        digits = ''
        for ch in piece:
            if ch.isdigit():
                digits += ch
            else:
                break
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _version_lt(got, minimum):
    a, b = version_tuple(got), version_tuple(minimum)
    n = max(len(a), len(b))
    a = a + (0,) * (n - len(a))
    b = b + (0,) * (n - len(b))
    return a < b


def evaluate(report, require):
    """Return a list of failure strings. Empty means the load check passed."""
    fails = []
    require = require or {}
    report = report or {}
    probe = report.get('probe') or {}
    exact = report.get('exact_stack') or {}
    loras = list(exact.get('loras') or [])
    by_name = {}
    for lo in loras:
        if lo.get('name'):
            by_name[lo['name']] = lo

    if 'max_lossy_merges' in require:
        if 'lossy_merges' not in probe:
            fails.append('lossy_merges missing')
        elif probe.get('lossy_merges') > require['max_lossy_merges']:
            fails.append(f"lossy merges {probe.get('lossy_merges')}")

    if require.get('no_load_warnings'):
        for lo in loras:
            warnings = lo.get('n_load_warnings') or lo.get('load_warnings') or lo.get('unsupported')
            if warnings:
                fails.append(
                    f"{lo.get('name')} warnings {list(lo.get('load_warnings') or [])[:3]} "
                    f"{list(lo.get('unsupported') or [])[:3]}"
                )

    for name, want in (require.get('expect') or {}).items():
        lo = by_name.get(name)
        if not lo:
            fails.append(f'{name} missing')
            continue
        got = (lo.get('adapters'), lo.get('dense_modules'))
        if got != tuple(want):
            fails.append(f"{name} adapters/dense {got[0]}/{got[1]} != {list(want)}")

    hooks = exact.get('hooks_by_type') or {}
    for kind, minimum in (require.get('min_hooks') or {}).items():
        bucket = hooks.get(kind) or {}
        got = sum(bucket.values()) if isinstance(bucket, dict) else 0
        if got < minimum:
            fails.append(f'hooks {kind} {got} < {minimum}')

    for name, strength in (require.get('strengths') or {}).items():
        lo = by_name.get(name)
        if not lo:
            fails.append(f'{name} missing for strength check')
            continue
        try:
            got = float(lo.get('strength'))
            want = float(strength)
        except (TypeError, ValueError):
            fails.append(f'{name} strength unreadable')
            continue
        if abs(got - want) > 1e-6:
            fails.append(f'{name} strength {got} != {want}')

    key = require.get('injection_key')
    if key:
        count = (probe.get('injection_keys') or {}).get(key) or 0
        if count <= 0:
            fails.append(f'injection {key} missing')

    minimum = require.get('min_comfyui')
    if minimum:
        got = report.get('comfyui')
        if not got or _version_lt(got, minimum):
            fails.append(f'comfyui {got} < {minimum}')
    return fails
