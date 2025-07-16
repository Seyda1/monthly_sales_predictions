import pymysql

conn = pymysql.connect(
    host='database-ntt.cohs2u2cme9j.us-east-1.rds.amazonaws.com',
    user='admin',
    password='Nosceteipsum1.',
    port=3306
)

try:
    with conn.cursor() as cur:
        cur.execute("CREATE DATABASE IF NOT EXISTS sales_predictions")
    conn.commit()
    print("Database created or already exists.")
finally:
    conn.close()
