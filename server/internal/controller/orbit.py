import math
from fastapi import APIRouter, Path, Depends
from internal.app.app import orbit_unit
from typing import List, Annotated, Optional
from pydantic import BaseModel
from internal.model.orbit import Orbit


class OrbitMap(BaseModel):
    id: int
    x: float
    z: float
    v: float


class OrbitMetaResponse(BaseModel):
    id: int
    x: float
    z: float
    v: float
    t: float
    ax: float
    ay: float
    az: float
    cj: float
    stable: Optional[bool]


class GetAllMetaResponse(BaseModel):
    meta: List[OrbitMetaResponse]
    next_page: Optional[int] = None


class GetByIDBody(BaseModel):
    id: int
    x: float
    z: float
    v: float
    t: float
    ax: float
    ay: float
    az: float
    cj: float
    stable: Optional[bool]
    tags: List[str]


class CreateOrbitJSON(BaseModel):
    x: float
    z: float
    v: float
    alpha: float
    t_period: float
    l1_r: float
    l2_r: float
    l3_r: float
    l4_r: float
    l5_r: float
    l6_r: float
    l1_im: float
    l2_im: float
    l3_im: float
    l4_im: float
    l5_im: float
    l6_im: float
    ax: float
    ay: float
    az: float
    dist_primary: float
    dist_secondary: float
    dist_curve: float
    cj: float
    stable: bool
    stability_order: int
    tags: list[str]


class GetByFamilyJSON(BaseModel):
    id: int
    t: float
    ax: float
    ay: float
    az: float
    dist_primary: float
    dist_secondary: float
    cj: float
    max_floke: float


orbits_router = APIRouter(
    prefix="/orbits", tags=["orbits"]
)


@orbits_router.post("", response_model=dict)
async def create_orbit(orbit: CreateOrbitJSON):
    orbit_id = orbit_unit.create_orbit(orbitJSON_to_orbit(orbit), orbit.tags)
    return {"id": orbit_id}


@orbits_router.get("/map", response_model=list[OrbitMap])
async def get_all_map(
        t_min: float = None, t_max: float = None, ax_min: float = None, ax_max: float = None, ay_min: float = None,
        ay_max: float = None, az_min: float = None, az_max: float = None, cj_min: float = None, cj_max: float = None,
        tag: str = None):
    filters = {
        "t_min": t_min, "t_max": t_max,
        "ax_min": ax_min, "ax_max": ax_max,
        "ay_min": ay_min, "ay_max": ay_max,
        "az_min": az_min, "az_max": az_max,
        "cj_min": cj_min, "cj_max": cj_max,
        "tag": tag
    }

    if t_min is not None and t_min < 0:
        filters["t_min"] = 0

    if t_max is not None and t_max < 1:
        return []

    if t_max is not None:
        filters["t_min"] = 1

    orbits = orbit_unit.get_all_map(filters)
    return orbits_to_map(orbits)


# return empty list if tag is not mentioned
@orbits_router.get("/meta", response_model=GetAllMetaResponse, response_model_exclude_unset=True)
async def get_all_meta(
        t_min: float = None, t_max: float = None, ax_min: float = None, ax_max: float = None, ay_min: float = None,
        ay_max: float = None, az_min: float = None, az_max: float = None, cj_min: float = None, cj_max: float = None,
        tag: str = None, page: int = 0):
    filters = {
        "t_min": t_min, "t_max": t_max,
        "ax_min": ax_min, "ax_max": ax_max,
        "ay_min": ay_min, "ay_max": ay_max,
        "az_min": az_min, "az_max": az_max,
        "cj_min": cj_min, "cj_max": cj_max,
        "tag": tag
    }

    orbits = orbit_unit.get_all_meta(filters, page)
    if len(orbits) > 0:
        return GetAllMetaResponse(meta=orbits_to_meta(orbits), next_page=page+1)

    return GetAllMetaResponse(meta=orbits_to_meta(orbits))


@orbits_router.get("/{idx}", response_model=GetByIDBody)
async def get_by_id(
        idx: Annotated[int, Path(title="orbit ID", ge=1)]):
    orbit, tags = orbit_unit.get_by_id(idx)
    return orbit_to_response(orbit, tags)


@orbits_router.get("/family/{family}", response_model=List[GetByFamilyJSON])
async def get_by_family(
        family: Annotated[str, Path(title="family name (tag)")]):
    orbits = orbit_unit.get_by_family(family)
    return orbits_to_get_family_json(orbits)


def orbits_to_meta(orbits: List[Orbit]):
    meta = list()

    for elem in orbits:
        meta.append(OrbitMetaResponse(
            id=elem.id, x=elem.x, z=elem.z, v=elem.v, t=elem.t_period,
            ax=elem.ax, ay=elem.ay, az=elem.az, cj=elem.cj, stable=elem.stable)
        )

    return meta


def orbit_to_response(orbit: Orbit, tags: List[str]):
    return dict({
        "id": orbit.id, "x": orbit.x, "z": orbit.z, "v": orbit.v,
        "t": orbit.t_period, "ax": orbit.ax, "ay": orbit.ay, "az": orbit.az, "cj": orbit.cj, "stable": orbit.stable,
        "tags": tags
        })


def orbits_to_map(orbits: List[Orbit]):
    meta = list()

    for elem in orbits:
        meta.append({
            "id": elem.id, "x": elem.x, "z": elem.z, "v": elem.v
        })

    return meta


def orbitJSON_to_orbit(orbit: CreateOrbitJSON) -> Orbit:
    o = Orbit()
    o.x = orbit.x
    o.z = orbit.z
    o.v = orbit.v
    o.alpha = orbit.alpha
    o.t_period = orbit.t_period
    o.l1_r = orbit.l1_r
    o.l2_r = orbit.l2_r
    o.l3_r = orbit.l3_r
    o.l4_r = orbit.l4_r
    o.l5_r = orbit.l5_r
    o.l6_r = orbit.l6_r
    o.l1_im = orbit.l1_im
    o.l2_im = orbit.l2_im
    o.l3_im = orbit.l3_im
    o.l4_im = orbit.l4_im
    o.l5_im = orbit.l5_im
    o.l6_im = orbit.l6_im
    o.ax = orbit.ax
    o.ay = orbit.ay
    o.az = orbit.az
    o.dist_primary = orbit.dist_primary
    o.dist_secondary = orbit.dist_secondary
    o.dist_curve = orbit.dist_curve
    o.cj = orbit.cj
    o.stable = orbit.stable
    o.stability_order = orbit.stability_order

    return o


def orbits_to_get_family_json(orbits):
    o = list()

    for elem in orbits:
        max_floke = 0
        floke = [math.sqrt(elem.l1_r**2 + elem.l1_im**2), math.sqrt(elem.l2_r**2 + elem.l2_im**2),
                 math.sqrt(elem.l3_r**2 + elem.l3_im**2), math.sqrt(elem.l4_r**2 + elem.l4_im**2),
                 math.sqrt(elem.l5_r**2 + elem.l5_im**2), math.sqrt(elem.l6_r**2 + elem.l6_im**2)]

        for f in floke:
            if f > max_floke:
                max_floke = f

        o.append(GetByFamilyJSON(
            id=elem.id, t=elem.t_period, ax=elem.ax, ay=elem.ay, az=elem.az, dist_primary=elem.dist_primary,
            dist_secondary=elem.dist_secondary, cj=elem.cj, max_floke=max_floke)
        )

    return o
