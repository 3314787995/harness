"""Viewpoint claims require distinct cameras, landmarks and explicit axes."""

from ..types import MissingCapability


def viewpoint(value, frames):
    camera = frames[value["camera_frame"]]
    observer = frames[value["observer_frame"]]
    if len(set(value["landmark_ids"])) < 2:
        raise MissingCapability(
            "viewpoint", "viewpoint comparison needs multiple independent landmarks"
        )
    if camera["kind"] == "image" or observer["kind"] == "image":
        raise MissingCapability(
            "reference_frame", "image y-coordinate does not establish camera height"
        )
    if not value["axis_defined"]:
        raise MissingCapability(
            "reference_frame", "viewpoint angle/height lacks explicit observer and axes"
        )
    if camera["kind"] == "mirror" or observer["kind"] == "mirror":
        mirror = frames.get(value["mirror_plane_id"])
        if not mirror or mirror["kind"] != "mirror":
            raise MissingCapability(
                "viewpoint", "mirror relation requires a separately identified mirror plane"
            )
    if camera["kind"] == "inner_camera" and not camera["parent_frame_id"]:
        raise MissingCapability(
            "viewpoint", "screen camera requires a parent screen/outer-view mapping"
        )
    return value["relation"]
