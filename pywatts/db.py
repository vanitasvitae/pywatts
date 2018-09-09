import pandas as pd
from peewee import *
from playhouse import sqlite_ext
from playhouse.sqlite_ext import SqliteExtDatabase

import os.path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(BASE_DIR, "pywatts.db")
db = SqliteExtDatabase(db_path)


class WeatherStation(Model):
    longitude = IntegerField()
    latitude = IntegerField()
    location = CharField()
    elevation = IntegerField()
    city = CharField()
    state = CharField()
    id = CharField(unique=True)

    class Meta:
        database = db


class Result(Model):
    station = ForeignKeyField(WeatherStation)
    dc_output = sqlite_ext.JSONField()
    temperature = sqlite_ext.JSONField()
    wind_speed = sqlite_ext.JSONField()

    class Meta:
        database = db


def rows_to_df(indices):
    temps = []
    dcs = []
    winds = []

    db.connect()

    for result in Result.select().where(Result.id << indices):
        temps += result.temperature
        dcs += result.dc_output
        winds += result.wind_speed

    db.close()

    return pd.DataFrame(
        {'temp': temps,
         'dc': dcs,
         'wind': winds
         })
