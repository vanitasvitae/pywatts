from pypvwatts import PVWatts
from pywatts.db import *
from pathlib import Path



def fetch_data(from_lon, to_lon, from_lat, to_lat, step_size=1):
    my_api_key = ''
    with open(str(Path.home()) + '/pvwatts_api_key.txt', 'r') as file:
        my_api_key = file.readline()
        # Strip newline
        my_api_key = my_api_key[:-1]

    p = PVWatts(api_key=my_api_key)

    results = []

    for longitude in range(from_lon, to_lon, step_size):
        for latitude in range(from_lat, to_lat, step_size):
            try:
                results.append(p.request(
                    format='JSON', system_capacity=4, module_type=1, array_type=1,
                    azimuth=190, tilt=40, dataset='intl', timeframe='hourly',
                    losses=10, lon=longitude, lat=latitude
                ))
            except:
                print("Could not fetch all data")
                return results

    return results


def store_data(result_list):
    db.connect()

    if not db.table_exists(WeatherStation):
        db.create_tables([WeatherStation, Result])

    for result in result_list:
        if result.errors:
            continue

        if WeatherStation.select().where(WeatherStation.id == result.station_info['location']).exists():
            print("Data for station %s already in database. Skipping." % result.station_info['location'])
            continue

        station = WeatherStation.create(
            longitude=result.station_info['lon'],
            latitude=result.station_info['lat'],
            location=result.station_info['location'],
            elevation=result.station_info['elev'],
            city=result.station_info['city'],
            state=result.station_info['state'],
            id=result.station_info['location'],
        )

        Result.create(
            station=station,
            dc_output=result.outputs['dc'],
            temperature=result.outputs['tamb'],
            wind_speed=result.outputs['wspd']
        )

    db.close()
