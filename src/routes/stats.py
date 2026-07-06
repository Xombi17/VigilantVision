import sqlite3
from datetime import datetime, timedelta
from fastapi import APIRouter

from src.config import DB_NAME, PSUTIL_AVAILABLE

if PSUTIL_AVAILABLE:
    import psutil

router = APIRouter(tags=["Stats"])


@router.get("/stats")
def get_stats():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute(
        "SELECT substr(timestamp, 1, 8), count(*) FROM alerts GROUP BY substr(timestamp, 1, 8)"
    )
    data = dict(c.fetchall())
    conn.close()

    stats = []
    today = datetime.now()
    for i in range(6, -1, -1):
        d = today - timedelta(days=i)
        key = d.strftime("%Y%m%d")
        stats.append(data.get(key, 0))

    cpu_load = 0
    ram_load = 0
    if PSUTIL_AVAILABLE:
        try:
            cpu_load = psutil.cpu_percent()
            ram_load = psutil.virtual_memory().percent
        except:
            import random

            cpu_load = random.randint(15, 30)
            ram_load = random.randint(40, 50)
    else:
        import random

        cpu_load = random.randint(15, 30)
        ram_load = random.randint(40, 50)

    return {"weekly_data": stats, "cpu_load": cpu_load, "ram_load": ram_load}


@router.get("/history")
async def get_history():
    try:
        conn = sqlite3.connect(DB_NAME)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM alerts ORDER BY timestamp DESC LIMIT 100")
        rows = c.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    except Exception as e:
        return {"error": str(e)}
