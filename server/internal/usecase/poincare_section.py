from internal.db import PoincareSectionRepo, OrbitRepo, OrbitFamilyRepo
from internal.model.poincare_section import PoincareSection
from typing import List


class PoincareSectionUnit:
    poincare_section_repo: PoincareSectionRepo
    orbit_repo: OrbitRepo
    orbit_family_repo: OrbitFamilyRepo

    def __init__(self,
                 poincare_section_repo: PoincareSectionRepo, orbit_repo: OrbitRepo, orbit_family_repo: OrbitFamilyRepo):
        self.poincare_section_repo = poincare_section_repo
        self.orbit_repo = orbit_repo
        self.orbit_family_repo = orbit_family_repo

    def get_by_pk(self, idx: int, plane: str):
        return self.poincare_section_repo.get_by_pk(idx, plane)

    def create_poincare_section(self, orbit_id: int, plane: str, v: List[dict]):
        t = PoincareSection(orbit_id=orbit_id, plane=plane, v=v)
        self.poincare_section_repo.create_poincare_section(t)

    def get_by_cj(self, libration_point, cj_min, cj_max, plane):
        ids = self.orbit_family_repo.get_ids_by_libration_point(libration_point)
        ids_int = []
        for elem in ids:
            ids_int.append(elem.orbit_id)

        orbit_ids = self.orbit_repo.get_ids_by_cj(ids_int, cj_min, cj_max)
        ids_int = []
        for elem in orbit_ids:
            ids_int.append(elem.id)

        return self.poincare_section_repo.get_by_ids(ids_int, plane)
