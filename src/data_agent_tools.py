import fastf1
from typing import Literal, List
from smolagents import tool


class FastF1DataError(Exception):
    """Custom exception for FastF1 data retrieval errors."""
    pass

@tool
def get_lap(year: int, grand_prix: str, session_type: str, driver_code: str, lap_number: int = 0, output_path: str = "lap.parquet") -> str:
    """
    Retrieve a specific lap data for a specific driver in a session.

    Args:
        year (int): The season year (e.g. 2025)
        grand_prix (str): The name of the Grand Prix (e.g. "Miami Grand Prix")
        session_type (str): Session type: "FP1", "FP2", "Q", "R", etc.
        driver_code (str): The 3-letter driver code (e.g., 'VER')
        lap_number (int): The lap number (e.g. 20), if lap number is not specified it gets the fastest lap
        output_path (str): Output path of the parquet file containing the lap data. (e.g. "lap.parquet")

    Returns:
        str: Path of the parquet file containing the fastest lap

    Raises:
        FastF1DataError: If no laps are found for the driver
    """
    try:
        session = fastf1.get_session(year, grand_prix, session_type)
        session.load()
    except Exception as e:
        raise FastF1DataError(f"Failed to load session {year} - {grand_prix} ({session_type}): {e}")

    laps = session.laps
    driver_laps = laps.pick_drivers(driver_code)

    if driver_laps.empty:
        raise FastF1DataError(f"No laps found for driver '{driver_code}' in session.")

    if lap_number == 0 or lap_number is None:
        lap = driver_laps.pick_fastest()
    else:
        lap = driver_laps.pick_laps(lap_number)

    lap.drop("Time")
    #Parquet file
    lap.to_frame().T.to_parquet(output_path)

    return output_path

@tool
def get_all_laps(year: int, grand_prix: str, session_type: str, driver_code: str, output_path: str = "laps.parquet") -> str:
    """
    Retrieve all laps for a specific driver in a session.

    Args:
        year (int): The season year (e.g. 2025)
        grand_prix (str): The name of the Grand Prix (e.g. "Miami Grand Prix")
        session_type (str): Session type: "FP1", "FP2", "Q", "R", etc.
        driver_code (str): The 3-letter driver code (e.g., 'VER')
        output_path (str): Output path of the parquet file containing the lap data. (e.g. "laps.parquet")

    Returns:
        str: Path of the parquet file containing the fastest lap

    Raises:
        FastF1DataError: If no laps are found for the driver
    """
    try:
        session = fastf1.get_session(year, grand_prix, session_type)
        session.load()
    except Exception as e:
        raise FastF1DataError(f"Failed to load session {year} - {grand_prix} ({session_type}): {e}")

    laps = session.laps
    driver_laps = laps.pick_drivers(driver_code)

    if driver_laps.empty:
        raise FastF1DataError(f"No laps found for driver '{driver_code}' in session.")

    driver_laps = driver_laps.drop(columns=["Time","PitOutTime","PitInTime","Sector1SessionTime","Sector2SessionTime",
                                            "Sector3SessionTime","LapStartTime","LapStartDate","FastF1Generated","IsAccurate"])
    #Parquet file
    driver_laps.to_parquet(output_path)

    return output_path
@tool
def get_fastest_lap(year: int, grand_prix: str, session_type: str, driver_code: str, output_path: str = "fastest_lap.parquet") -> str:
    """
    Retrieve the fastest lap data for a specific driver in a session.

    Args:
        year (int): The season year (e.g. 2025)
        grand_prix (str): The name of the Grand Prix (e.g. "Miami Grand Prix")
        session_type (str): Session type: "FP1", "FP2", "Q", "R", etc.
        driver_code (str): The 3-letter driver code (e.g., 'VER')
        output_path (str): Output path of the parquet file containing the lap data. (e.g. "fastest_lap.parquet")

    Returns:
        str: Path of the parquet file containing the fastest lap

    Raises:
        FastF1DataError: If no laps are found for the driver
    """
    try:
        session = fastf1.get_session(year, grand_prix, session_type)
        session.load()
    except Exception as e:
        raise FastF1DataError(f"Failed to load session {year} - {grand_prix} ({session_type}): {e}")

    laps = session.laps
    driver_laps = laps.pick_drivers(driver_code)

    if driver_laps.empty:
        raise FastF1DataError(f"No laps found for driver '{driver_code}' in session.")

    lap = driver_laps.pick_fastest()

    lap = lap.drop("Time")
    #Parquet file
    lap.to_frame().T.to_parquet(output_path)

    return output_path
@tool
def get_telemetry_lap(year: int, grand_prix: str, session_type: str, driver_code: str, lap_number: int = 0,  output_path: str = "fastest_lap.parquet")-> str:
    """
    Retrieve telemetry data for a given lap of a driver in a session. If no lap_number argument is given, it returns the telemetry for the fastest lap.

    Args:
        year (int): The season year (e.g. 2025)
        grand_prix (str): The name of the Grand Prix (e.g. "Miami Grand Prix")
        session_type (str): Session type: "FP1", "FP2", "Q", "R", etc.
        driver_code (str): The 3-letter driver code (e.g., 'VER')
        lap_number (int): The lap number (e.g. 20), if lap number is not specified it gets the fastest lap
        output_path (str): Output path of the parquet file containing the telemetry data. (e.g. "telemetry.parquet")

    Returns:
        str: Path of the parquet file containing the telemetry

    Raises:
        FastF1DataError: If no laps are found for the driver
    """

    try:
        session = fastf1.get_session(year, grand_prix, session_type)
        session.load()
    except Exception as e:
        raise FastF1DataError(f"Failed to load session {year} - {grand_prix} ({session_type}): {e}")

    laps = session.laps
    driver_laps = laps.pick_drivers(driver_code)

    if driver_laps.empty:
        raise FastF1DataError(f"No laps found for driver '{driver_code}' in session.")

    if(lap_number == 0 or lap_number is None):
        lap = driver_laps.pick_fastest()
    else:
        lap = driver_laps.pick_laps(lap_number)

    try:
        telemetry = lap.get_car_data().add_distance()
        if telemetry.empty:
            raise ValueError("Telemetry is empty.")
        telemetry["LapTime"] = lap["LapTime"].iloc[0]
        telemetry["Team"] = lap["Team"].iloc[0]
        telemetry.to_parquet(output_path)
        return output_path
    except Exception as e:
        raise FastF1DataError(f"Failed to retrieve telemetry: {e}")

@tool
def get_telemetry_fastest_lap(year: int, grand_prix: str, session_type: str, driver_code: str, output_path: str = "fastest_lap.parquet")-> str:
    """
    Retrieve telemetry data for the fastest lap of a driver in a session. If no lap_number argument is given, it returns the telemetry for the fastest lap.

    Args:
        year (int): The season year (e.g. 2025)
        grand_prix (str): The name of the Grand Prix (e.g. "Miami Grand Prix")
        session_type (str): Session type: "FP1", "FP2", "Q", "R", etc.
        driver_code (str): The 3-letter driver code (e.g., 'VER')
        output_path (str): Output path of the parquet file containing the telemetry data. (e.g. "telemetry_fastest.parquet")

    Returns:
        str: Path of the parquet file containing the telemetry

    Raises:
        FastF1DataError: If no laps are found for the driver
    """

    try:
        session = fastf1.get_session(year, grand_prix, session_type)
        session.load()
    except Exception as e:
        raise FastF1DataError(f"Failed to load session {year} - {grand_prix} ({session_type}): {e}")

    laps = session.laps
    driver_laps = laps.pick_drivers(driver_code)

    if driver_laps.empty:
        raise FastF1DataError(f"No laps found for driver '{driver_code}' in session.")

    lap = driver_laps.pick_fastest()

    try:
        telemetry = lap.get_car_data().add_distance()
        if telemetry.empty:
            raise ValueError("Telemetry is empty.")
        telemetry["LapTime"] = lap["LapTime"]
        telemetry["Team"] = lap["Team"]
        telemetry.to_parquet(output_path)
        return output_path
    except Exception as e:
        raise FastF1DataError(f"Failed to retrieve telemetry: {e}")

@tool
def get_session_results(year: int, gp: str, session_type: str, output_path: str = "race_results.parquet",
                        columns: List[Literal["DriverNumber", "FullName", "TeamName", "TeamColor",
                        "CountryCode", "Position", "ClassifiedPosition", "GridPosition", "Q1", "Q2", "Q3",
                        "Time", "Status", "Points"]] | None = None) -> str:
    """
    Get final results from an F1 session and (if given by parameter) filter the results
      by drivers and by column.

    Args:
        year (int): Year of the Grand Prix.
        gp (str): Official name of the Grand Prix (e.g., "Spanish Grand Prix").
        session_type (str): Session type: "Q", "R", "FP1", etc.
        output_path (str): Output path of the parquet file containing the session results data. (e.g. "race_results.parquet")
        columns (List[str]): A list of column names the user wants to include in the output representing the following:
            DriverNumber | str | The number associated with this driver in this session (usually the drivers permanent number)
            FullName | str | The drivers full name (e.g. “Pierre Gasly”)
            Abbreviation | str | The drivers three letter abbreviation (e.g. “GAS”)
            TeamName | str | The team name (short version without title sponsors)
            TeamColor | str | The color commonly associated with this team (hex value)
            CountryCode | str | The driver’s country code (e.g. “FRA”)
            Position | float | The drivers finishing position (values only given if session is ‘Race’, ‘Qualifying’, ‘Sprint Shootout’, ‘Sprint’, or ‘Sprint Qualifying’).
            ClassifiedPosition | str | The official classification result for each driver. This is either an integer value if the driver is officially classified or one of “R” (retired), “D” (disqualified), “E” (excluded), “W” (withdrawn), “F” (failed to qualify) or “N” (not classified).
            GridPosition | float | The drivers starting position (values only given if session is ‘Race’, ‘Sprint’, ‘Sprint Shootout’ or ‘Sprint Qualifying’)
            Q1 | pd.Timedelta | The drivers best Q1 time (values only given if session is ‘Qualifying’ or ‘Sprint Shootout’)
            Q2 | pd.Timedelta | The drivers best Q2 time (values only given if session is ‘Qualifying’ or ‘Sprint Shootout’)
            Q3 | pd.Timedelta | The drivers best Q3 time (values only given if session is ‘Qualifying’ or ‘Sprint Shootout’)
            Time | pd.Timedelta | The drivers total race time (values only given if session is ‘Race’, ‘Sprint’, ‘Sprint Shootout’ or ‘Sprint Qualifying’ and the driver was not more than one lap behind the leader)
            Status | str | A status message to indicate if and how the driver finished the race or to indicate the cause of a DNF. Possible values include but are not limited to ‘Finished’, ‘+ 1 Lap’, ‘Crash’, ‘Gearbox’, … (values only given if session is ‘Race’, ‘Sprint’, ‘Sprint Shootout’ or ‘Sprint Qualifying’)
            Points | float | The number of points received by each driver for their finishing result.

    Returns:
        str: Path of the parquet file containing the race results
    """
    try:
        session = fastf1.get_session(year, gp, session_type)
        session.load()
    except Exception as e:
        raise FastF1DataError(f"Failed to load session {year} - {gp} ({session_type}): {e}")

    session_results = session.results

    if (columns is not None and len(columns) > 0):
        session_results = session_results[columns]

    session_results.to_parquet(output_path)
    return output_path
