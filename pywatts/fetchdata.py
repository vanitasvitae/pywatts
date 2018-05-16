from pypvwatts import PVWatts
from pywatts.models import *
from pathlib import Path


db = SqliteDatabase('pywatts.db')


def fetch_data(from_lon, to_lon, from_lat, to_lat, step_size=1):
    my_api_key = ''
    with open(str(Path.home()) + '/pvwatts_api_key.txt', 'r') as file:
        my_api_key = file.readline()
        # Strip newline
        my_api_key = my_api_key[:-1]

    p = PVWatts(api_key=my_api_key)

    results = []

    print(range(from_lon, to_lon))

    for longitude in range(from_lon, to_lon, step_size):
        for latitude in range(from_lat, to_lat, step_size):
            results.append(p.request(
                format='JSON', system_capacity=4, module_type=1, array_type=1,
                azimuth=190, tilt=40, dataset='intl', timeframe='hourly',
                losses=10, lon=longitude, lat=latitude
            ))

    return results


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
