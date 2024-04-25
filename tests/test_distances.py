"""Test file for the Script/distances.py file."""
# pylint: disable=wrong-import-position
# pylint: disable=import-error
# autopep8 Args : --ignore=E402
# > These are all the modifications that we made to enable local imports


# Import the built-in modules
import sys
import os

# Import the third-party modules
import pandas as pd
import pytest

# Import the local modules
sys.path.append('../scripts/')
from featherflight import FeatherFlight
from distances import get_distance, get_point

# Testing the __init__ method of the FeatherFlight class


def test_constructor():
    """Make sure that the data was loaded correctly."""
    ff = FeatherFlight()
    assert isinstance(ff.airports, pd.DataFrame)
    assert isinstance(ff.routes, pd.DataFrame)
    assert isinstance(ff.airplanes, pd.DataFrame)
    assert isinstance(ff.airlines, pd.DataFrame)


ROUTES_INPUT = {
    "Source airport ID": ["3008", "1200", "2745"],
    "Destination airport ID": ["3020", "629", "3576"],
}


@pytest.fixture
def ff_distances_instance():
    """Fixture to create a FeatherFlight instance with three routes 
    and compute their distances."""
    ff = FeatherFlight()
    ff.routes = pd.DataFrame(ROUTES_INPUT)
    ff.calculate_distances_routes()
    return ff


def test_get_distance_short(ff_distances_instance):
    """Check that the distance between two airports is correct

    Parameters
    ----------
    ff_distances_instance : fixture
        The fixture that returns a FeatherFlight instance with three routes
        and their computed distances.
    """
    assert get_distance("3008", "3020", ff_distances_instance.airports) == pytest.approx(
        635, rel=0.01)


def test_get_distance_medium(ff_distances_instance):
    """Check that the distance between two airports is correct

    Parameters
    ----------
    ff_distances_instance : fixture
        The fixture that returns a FeatherFlight instance with three routes
        and their computed distances.
    """
    assert get_distance("1200", "629", ff_distances_instance.airports) == pytest.approx(
        1214, rel=0.01)


def test_get_distance_long(ff_distances_instance):
    """Check that the distance between two airports is correct

    Parameters
    ----------
    ff_distances_instance : fixture
        The fixture that returns a FeatherFlight instance with three routes
        and their computed distances.
    """
    assert get_distance("2745", "3576", ff_distances_instance.airports) == pytest.approx(
        2242, rel=0.01)


def test_distances_method_short(ff_distances_instance):
    """Check that the shortest route has the correct distance

    Parameters
    ----------
    ff_distances_instance : fixture
        The fixture that returns a FeatherFlight instance with three routes
        and their computed distances.
    """
    assert ff_distances_instance.routes.distance_km.iloc[0] == pytest.approx(
        635, rel=0.01)


def test_distances_method_medium(ff_distances_instance):
    """Check that the medium route has the correct distance

    Parameters
    ----------
    ff_distances_instance : fixture
        The fixture that returns a FeatherFlight instance with three routes
        and their computed distances.
    """
    assert ff_distances_instance.routes.distance_km.iloc[1] == pytest.approx(
        1214, rel=0.01)


def test_distances_method_long(ff_distances_instance):
    """Check that the longest route has the correct distance

    Parameters
    ----------
    ff_distances_instance : fixture
        The fixture that returns a FeatherFlight instance with three routes
        and their computed distances.
    """
    assert ff_distances_instance.routes.distance_km.iloc[2] == pytest.approx(
        2242, rel=0.01)
