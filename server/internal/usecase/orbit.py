from internal.model.orbit import Orbit
from internal.model.orbit_family import OrbitFamily
from internal.db import OrbitRepo, OrbitFamilyRepo
from typing import List


class OrbitUnit:
    orbit_repo: OrbitRepo
    orbit_family_repo: OrbitFamilyRepo

    def __init__(self, orbit_repo: OrbitRepo, orbit_family_repo: OrbitFamilyRepo):
        self.orbit_repo = orbit_repo
        self.orbit_family_repo = orbit_family_repo

    def create_orbit(self, orbit: Orbit, tags: list[str]) -> int:
        orbit_id = self.orbit_repo.create_orbit(orbit)
        tags_models = list()
        for tag in tags:
            tags_models.append(OrbitFamily(orbit_id=orbit_id, tag=tag))

        self.orbit_family_repo.create_tags_by_id(tags_models)

        return orbit_id

    def get_all_map(self, filters: dict):
        if filters["tag"] is not None:
            ids = self.orbit_family_repo.get_ids_by_tag(filters["tag"])

            ids_int = []
            for elem in ids:
                ids_int.append(elem.orbit_id)

            return self.orbit_repo.get_all_map(filters, ids_int)

        return self.orbit_repo.get_all_map(filters)

    def get_all_meta(self, filters: dict, page: int):
        ids = self.orbit_family_repo.get_paginated_ids_by_tag(filters["tag"], page)

        ids_int = []
        for elem in ids:
            ids_int.append(elem.orbit_id)

        return self.orbit_repo.get_all_meta(filters, ids_int)

    def get_by_id(self, idx: int):
        tags = self.orbit_family_repo.get_tags_by_orbit_id(idx)
        return self.orbit_repo.get_by_id(idx), tags

    def get_by_family(self, family: str):
        ids = self.orbit_family_repo.get_ids_by_tag(family)

        ids_int = []
        for elem in ids:
            ids_int.append(elem.orbit_id)

        return self.orbit_repo.get_by_family(ids_int)
