from fastapi import APIRouter, Path
from internal.app.app import libration_point_unit
from typing import Annotated


libration_points_router = APIRouter(
    prefix="/libration_points", tags=["libration points"]
)


# @libration_points_router.get("/{item_id}", response_model=LibrationPoint)
@libration_points_router.get("/{item_id}")
async def get(item_id: Annotated[int, Path(title="Libration point ID", ge=1)]):
    libration_point = libration_point_unit.get_by_id(item_id)
    return {"id": libration_point.id, "name": libration_point.name}
