from fastapi import APIRouter, Path, Depends
from pydantic import BaseModel
from internal.app.app import poincare_section_unit
from internal.model.error import NotValidError
from typing import Annotated
from typing import List
from internal.controller.middleware import verify_token


poincare_sections_router = APIRouter(
    prefix="/poincare_sections", tags=["poincare_sections"]
)


class CreatePoincareSectionJSON(BaseModel):
    orbit_id: int
    plane: str
    v: List[dict]


class GetByCjJSON(BaseModel):
    orbit_id: int
    plane: str
    v: List[dict]


@poincare_sections_router.get("/{idx}/{plane}", response_model=List[dict])
async def get_by_id(
        idx: Annotated[int, Path(title="Poincare section id", ge=1)],
        plane: str
):
    return poincare_section_unit.get_by_pk(idx, plane)


@poincare_sections_router.post("")
async def create_poincare_section(_: Annotated[str, Depends(verify_token)], body: CreatePoincareSectionJSON):
    validation_create_poincare_section(body.v)
    poincare_section_unit.create_poincare_section(body.orbit_id, body.plane, body.v)


@poincare_sections_router.get("/cj", response_model=List[GetByCjJSON])
async def get_by_cj(
        libration_point: str,
        cj: float,
        d: float,
        plane: str
):
    sections = poincare_section_unit.get_by_cj(libration_point, cj-d, cj+d, plane)
    return sections_to_get_by_cj_json(sections)


def sections_to_get_by_cj_json(sections):
    body = []
    for elem in sections:
        body.append(GetByCjJSON(orbit_id=elem.orbit_id, plane=elem.plane, v=elem.v))

    return body


def validation_create_poincare_section(v: List[dict]):
    if len(v) < 1:
        raise NotValidError

    # t0 = v[0]
    # if "x" not in t0 or "y" not in t0 or "z" not in t0 or "vx" not in t0 or "vy" not in t0 or "vz" not in t0:
    #     raise NotValidError
