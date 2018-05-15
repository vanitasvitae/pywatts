from pypvwatts import PVWatts
from .models import *
from pathlib import Path


db = SqliteDatabase('pywatts.db')


def fetch_data(from_long, to_long, from_lat, to_lat, step_size=1):
    my_api_key = ''
    with open(str(Path.home()) + '/pvwatts_api_key.txt', 'r') as file:
        api_key = file.readline()

    p = PVWatts(api_key=my_api_key)

    result_list = []

    for longitude in range(from_long, to_long, step_size):
        for latitude in range(from_lat, to_lat, step_size):
            result_list.append(p.request(
                system_capacity=4, module_type=1, array_type=1, format='json',
                azimuth=190, tilt=30, dataset='intl', timeframe='hourly',
                losses=13, lon=longitude, lat=latitude
            ))

    return result_list


def store_data(db_name, result_list):
    db.connect()
    db.create_tables([WeatherStation, Result])

    for result in result_list:
        if WeatherStation.select().where(WeatherStation.id == result.location).exists():
            print("Data for station %s already in database. Skipping." % result.station_info.location)
            continue

        station = WeatherStation.create(
            longitude=result.station_info.lon,
            latitude=result.station_info.lat,
            location=result.station_info.location,
            elevation=result.station_info.elevation,
            city=result.station_info.city,
            state=result.station_info.state,
            id=result.station_info.location,
        )

        Result.create(
            station=station,
            dc_output=result.outputs.dc,
            temperature=result.outputs.tamb,
            wind_speed=result.outputs.wspd
        )
