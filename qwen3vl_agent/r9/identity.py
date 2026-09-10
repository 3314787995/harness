"""Identity updates are transactional with their source observations."""

from .spatial_state import SpatialState


def update_entity_links(state: SpatialState, payload, evidence, namespace, config):
    return state.observe(payload, evidence, namespace, config.max_active_binding_hypotheses)
