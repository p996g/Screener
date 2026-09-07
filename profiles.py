"""
Resolves named screen profiles from settings.yaml into fully-merged run
configs.

A profile (settings.yaml's `profiles` section) is a named, *partial*
override of the top-level config: it lists only what it wants to change —
usually its own `filters` and `ranking` — and inherits everything else
(universe, data, any filter parameter it doesn't mention, output) from the
top-level sections unchanged. This is what lets several very different
screens share one universe and one liquidity floor without repeating them,
while still letting a profile override just *one field* of a shared
filter's config (e.g. use a 200-day trend for a pullback screen while the
top-level default stays 50-day) rather than having to redeclare the whole
block.

Adding a new profile is pure configuration: add a new key under `profiles`
in settings.yaml with whatever it needs to override. Nothing here changes.
"""

# The top-level sections a profile is allowed to override. universe/data
# are included so a profile *can* screen a different ticker list or pull
# more history if it needs to, even though none of the built-in example
# profiles do — they only touch filters/ranking/output.
GLOBAL_SECTIONS = ["universe", "data", "filters", "ranking", "output", "data_quality"]


def list_profile_names(config: dict) -> list[str]:
    """Names of every profile defined in settings.yaml's `profiles`
    section, in file order."""
    return list((config.get("profiles") or {}).keys())


def resolve_profile_config(config: dict, profile_name: str) -> dict:
    """Build the fully-resolved run config for one profile: each of
    universe/data/filters/ranking/output is the top-level section deep-
    merged with that profile's own override of the same section (if any).

    Raises ValueError if `profile_name` isn't defined in `profiles`.
    """
    profiles = config.get("profiles") or {}
    if profile_name not in profiles:
        available = ", ".join(profiles) or "(none defined)"
        raise ValueError(
            f"Unknown profile '{profile_name}' — defined profiles: {available}"
        )

    overrides = profiles[profile_name] or {}
    return {
        section: _deep_merge(config.get(section, {}), overrides.get(section, {}))
        for section in GLOBAL_SECTIONS
    }


def _deep_merge(base, override):
    """Recursively merge `override` onto `base`.

    Dict values are merged key-by-key, so a profile can override a single
    field deep inside a filter's config (or a single ranking weight) and
    inherit every sibling field untouched. Anything else (numbers,
    strings, lists, None) is replaced wholesale by the override — the
    override always wins once we're below the dict layer.
    """
    if not isinstance(base, dict) or not isinstance(override, dict):
        return base if override is None else override

    merged = dict(base)
    for key, value in override.items():
        merged[key] = _deep_merge(merged.get(key), value)
    return merged
