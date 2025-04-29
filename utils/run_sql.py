import os
import psycopg2
import pandas as pd

def db_connect():
    DB_HOST = os.getenv('CLIMATETRACE_HOST')
    DB_PORT = os.getenv('CLIMATETRACE_PORT')
    DB_USER = os.getenv('CLIMATETRACE_USER')
    DB_PASS = os.getenv('CLIMATETRACE_PASS')
    DB_NAME = os.getenv('CLIMATETRACE_DB')

    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS
    )

    return conn


def run_sql(query):

    conn = db_connect()
    
    try:
        result = pd.read_sql(query, conn)

    finally:
        conn.close()
   
    return result

