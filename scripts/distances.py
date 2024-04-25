"""Compute the distance between two airports."""

from geopy.distance import geodesic
import pandas as pd


def get_point(airport_id: str, df: pd.DataFrame) -> list:
    """Get the [latitude, longitude] of a given airport.

    Parameters
    ----------
    airport_id : str
        Identifier of the airport.
    df : pandas.DataFrame
        DataFrame containing the airports data.

    Returns
    -------
    list
        Latitude and longitude of the airport. If the airport is not found, return None.
    """
    try:
        return [df[df["Airport ID"] == airport_id].Latitude.iloc[0],
                df[df["Airport ID"] == airport_id].Longitude.iloc[0]]
    except IndexError:
        return None


def get_distance(airport1: str, airport2: str, df: pd.DataFrame) -> float:
    """Get the distance between two airports.

    Parameters
    ----------
    airport1 : str
        Identifier of the first airport.
    airport2 : str
        Identifier of the second airport.
    df : pandas.DataFrame
        DataFrame containing the airports data.

    Returns
    -------
    float
        Distance between the two airports in kilometers. If the airports are not found, return None.
    """
    try:
        return geodesic(get_point(airport1, df), get_point(airport2, df)).kilometers
    except TypeError:
        return None
