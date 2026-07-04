"""
Backward-compatibility shim: GlobalRoutePlannerDAO
In CARLA 0.9.16 this class was merged into GlobalRoutePlanner.
We provide a lightweight wrapper so older code still works.
"""


class GlobalRoutePlannerDAO:
    """Compatibility shim — stores map and resolution for GlobalRoutePlanner."""

    def __init__(self, wmap, sampling_resolution=2.0):
        self._map = wmap
        self._sampling_resolution = sampling_resolution

    def get_map(self):
        return self._map

    def get_sampling_resolution(self):
        return self._sampling_resolution
