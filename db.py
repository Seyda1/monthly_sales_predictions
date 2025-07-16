# db.py
import pymysql
from settings import settings

conn = pymysql.connect(
    host=settings.DB_HOST,
    user=settings.DB_USER,
    password=settings.DB_PASSWORD,
    db=settings.DB_NAME,
    port=settings.DB_PORT
)

def save_prediction_to_db(shop_id: int, item_id: int, next_month: str, preds: float):
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                shop_id INT,
                item_id INT,
                next_month DATE,
                preds FLOAT
            )
        """)
        cur.execute(
            "INSERT INTO predictions (shop_id, item_id, next_month, preds) VALUES (%s, %s, %s, %s)",
            (shop_id, item_id, next_month, preds)
        )
        conn.commit()
