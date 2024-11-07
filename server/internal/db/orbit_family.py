from sqlalchemy.engine.base import Engine
from sqlalchemy.exc import SQLAlchemyError
from internal.model.orbit_family import OrbitFamily
from internal.model.error import NotFoundError
from sqlalchemy.orm import Session
from typing import List


class OrbitFamilyRepo:
    engine: Engine

    def __init__(self, engine: Engine):
        self.engine = engine

    def create_tags_by_id(self, tags):
        with Session(self.engine) as session:
            try:
                session.add_all(tags)
                session.commit()
            except SQLAlchemyError as e:
                session.rollback()
                raise RuntimeError(f"Database error occurred: {e}")

    def get_ids_by_tag(self, tag: str) -> List[OrbitFamily]:
        with Session(self.engine) as session:
            orbit_families = session.query(OrbitFamily).with_entities(OrbitFamily.orbit_id).filter(
                OrbitFamily.tag == tag
            ).all()

            if not orbit_families:
                return []

        return orbit_families

    def get_paginated_ids_by_tag(self, tag: str, page: int) -> List[OrbitFamily]:
        limit = int(50)
        with Session(self.engine) as session:
            orbit_families = session.query(OrbitFamily).with_entities(OrbitFamily.orbit_id)

            if tag is not None:
                orbit_families = orbit_families.filter(OrbitFamily.tag == tag)

            orbit_families = orbit_families.limit(limit).offset(page*limit)

            orbit_families = orbit_families.all()

        return orbit_families

    def get_tags_by_orbit_id(self, idx: int) -> List[str]:
        with Session(self.engine) as session:
            tags = session.query(OrbitFamily).with_entities(OrbitFamily.tag).filter(
                OrbitFamily.orbit_id == idx
            ).all()

            if not tags:
                return []

        return [tag.tag for tag in tags]

    def get_ids_by_libration_point(self, libration_point):
        with Session(self.engine) as session:
            orbit_families = session.query(OrbitFamily).with_entities(OrbitFamily.orbit_id).filter(
                OrbitFamily.tag.like(f"{libration_point}%")
            ).all()

            if not orbit_families:
                return []

        return orbit_families


# if __name__ == "__main__":
#     from internal.app.app import db
#     o_r = OrbitFamilyRepo(db.engine)
#     orbits = o_r.get_tags_by_orbit_id(
#         1
#     )
#
#     print(orbits)
