from dotenv import load_dotenv
from sqlalchemy import create_engine
import pandas as pd

# load the .env file variables
load_dotenv()

from sqlalchemy import create_engine

engine = create_engine('sqlite:///Proyecto_Data_Science_VTIindex.db')


def db_connect():
    import os
    engine = create_engine(os.getenv('DATABASE_URL'))
    engine.connect()
    return engine
