"""Working observation views contain target descriptions and neutral frame IDs."""

from .compiler import observation_task


def observation_payload(spec, state, action, evidence):
    context = state.context()
    return {
        "task": observation_task(spec),
        "acquisition": action,
        "frames": {
            alias: {
                k: e[k]
                for k in ("id", "source_frame_id", "timestamp_seconds", "source_size", "view_box")
            }
            for alias, e in evidence.items()
        },
        "known_observations": context["observations"],
        "known_links": context["entity_links"],
    }
