from peewee import *
from playhouse import sqlite_ext
from playhouse.sqlite_ext import SqliteExtDatabase

db = SqliteExtDatabase('pywatts.db')


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

