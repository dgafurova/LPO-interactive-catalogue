from internal.db.trajectory import TrajectoryRepo
from internal.model.trajectory import Trajectory
from typing import List


class TrajectoryUnit:
    trajectory_repo: TrajectoryRepo

    def __init__(self, trajectory_repo: TrajectoryRepo):
        self.trajectory_repo = trajectory_repo

    def get_by_id(self, orbit_id: int) -> List[dict]:
        return self.trajectory_repo.get_by_id(orbit_id)

    def get_nearest(self, x: float, y: float, z: float, vx: float, vy: float, vz: float):
        return self.trajectory_repo.get_nearest(x, y, z, vx, vy, vz)

    def create_trajectory(self, orbit_id: int, v: List[dict]):
        t = Trajectory(orbit_id=orbit_id, v=v, x=v[0]["x"], y=v[0]["y"], z=v[0]["z"],
                       vx=v[0]["vx"], vy=v[0]["vy"], vz=v[0]["vz"])
        self.trajectory_repo.create_trajectory(t)
