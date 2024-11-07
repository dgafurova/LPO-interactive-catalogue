from fastapi import APIRouter
from internal.controller import libration_points_router, orbits_router, trajectories_router, poincare_sections_router


router = APIRouter()
router_list = [libration_points_router, orbits_router, trajectories_router, poincare_sections_router]

for elem in router_list:
    # elem.tags = router.tags.append("/api/v1")
    router.include_router(elem)
