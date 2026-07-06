from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.camera import camera_manager

router = APIRouter(prefix="/cameras", tags=["Cameras"])


class CameraInput(BaseModel):
    name: str
    source: str


@router.post("")
async def add_new_camera(cam: CameraInput):
    result = camera_manager.add_camera(cam.source, cam.name)
    if result["id"]:
        with camera_manager.lock:
            cam_data = camera_manager.cameras.get(result["id"])
            cam_details = (
                {
                    "id": result["id"],
                    "name": cam_data["name"] if cam_data else cam.name,
                    "source": cam_data["source"] if cam_data else cam.source,
                    "status": "active",
                }
                if cam_data
                else None
            )
        return {"message": "Camera added", "camera": cam_details}
    else:
        raise HTTPException(status_code=400, detail="Failed to open camera")


@router.get("")
async def list_cameras():
    return camera_manager.get_active_cameras()


@router.delete("/{camera_id}")
async def delete_camera(camera_id: str):
    if camera_manager.remove_camera(camera_id):
        return {"message": "Camera removed"}
    raise HTTPException(status_code=404, detail="Camera not found")


@router.post("/{camera_id}/roi")
async def save_camera_roi(camera_id: str, data: dict):
    if "points" in data:
        points = data["points"]
        with camera_manager.lock:
            if camera_id in camera_manager.cameras:
                camera_manager.cameras[camera_id]["roi_points"] = points
                camera_manager.save_cameras()
                return {"status": "success", "roi_points": points}
        raise HTTPException(status_code=404, detail="Camera not found")


@router.get("/{camera_id}/roi")
async def get_camera_roi(camera_id: str):
    with camera_manager.lock:
        if camera_id in camera_manager.cameras:
            return {"points": camera_manager.cameras[camera_id].get("roi_points", [])}
    raise HTTPException(status_code=404, detail="Camera not found")
