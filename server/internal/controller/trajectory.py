from fastapi import APIRouter, Path, Depends
from pydantic import BaseModel
from internal.app.app import trajectory_unit
from internal.model.error import NotValidError
from typing import Annotated, List
from internal.controller.middleware import verify_token


trajectories_router = APIRouter(
    prefix="/trajectories", tags=["trajectories"]
)


class GetNearestJSON(BaseModel):
    orbit_id: int
    v: List[dict]


class CreateTrajectoryJSON(BaseModel):
    orbit_id: int
    v: List[dict]


@trajectories_router.get("/nearest", response_model=GetNearestJSON)
async def get_nearest(
        x: float, y: float, z: float, vx: float, vy: float, vz: float):
    t = trajectory_unit.get_nearest(x, y, z, vx, vy, vz)
    return GetNearestJSON(orbit_id=t[0], v=t[1])


@trajectories_router.get("/{orbit_id}", response_model=List[dict])
async def get_by_id(orbit_id: Annotated[int, Path(title="Orbit ID", ge=1)]):
    return trajectory_unit.get_by_id(orbit_id)


@trajectories_router.post("")
async def create_trajectory(_: Annotated[str, Depends(verify_token)], body: CreateTrajectoryJSON):
    validation_create_trajectory(body.v)
    trajectory_unit.create_trajectory(body.orbit_id, body.v)


def validation_create_trajectory(v: List[dict]):
    if len(v) < 1:
        raise NotValidError

    t0 = v[0]
    if "x" not in t0 or "y" not in t0 or "z" not in t0 or "vx" not in t0 or "vy" not in t0 or "vz" not in t0:
        raise NotValidError
