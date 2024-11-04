from internal.infra.db import DB
from internal.app.config import db_user, db_password, db_hostname, db_name, db_port
from internal.db import LibrationPointRepo, OrbitRepo, OrbitFamilyRepo, TrajectoryRepo, PoincareSectionRepo
from internal.usecase import LibrationPointUnit, OrbitUnit, TrajectoryUnit, PoincareSectionUnit

# db = DB(f'postgresql+psycopg2://{db_user}:{db_password}@{db_hostname}:{db_port}/{db_name}')
db = DB(f'postgresql://{db_user}:{db_password}@{db_hostname}:{db_port}/{db_name}')

libration_point_repo = LibrationPointRepo(db.engine)
orbit_repo = OrbitRepo(db.engine)
orbit_family_repo = OrbitFamilyRepo(db.engine)
trajectory_repo = TrajectoryRepo(db.engine)
poincare_section_repo = PoincareSectionRepo(db.engine)

libration_point_unit = LibrationPointUnit(libration_point_repo)
orbit_unit = OrbitUnit(orbit_repo, orbit_family_repo)
trajectory_unit = TrajectoryUnit(trajectory_repo)
poincare_section_unit = PoincareSectionUnit(poincare_section_repo, orbit_repo, orbit_family_repo)
