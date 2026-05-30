import io
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from PIL import Image
from app.services.experiment_service import latest_frame

router = APIRouter(prefix="/api", tags=["snapshot"])

@router.get("/sumo_snapshot.jpg")
def sumo_snapshot():
    global latest_frame
    if latest_frame is None:
        img = Image.new('RGB', (800, 500), color=(10, 17, 25))
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/jpeg")
    return StreamingResponse(io.BytesIO(latest_frame), media_type="image/jpeg")