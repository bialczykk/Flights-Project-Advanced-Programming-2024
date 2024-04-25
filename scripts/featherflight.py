# pylint: disable=no-name-in-module
# pylint: disable=no-self-argument
# pylint: disable=import-error
# pylint: disable=use-dict-literal

"""
This is the main module of the FeatherFlight class.
The FeatherFlight is a class that has been developed by the data team at FeatherFlight Inc. 
to get a better understanding of all our flights and drive the sustainable transition 
in our company.
"""

# Standard library imports
import zipfile
import io
import os
import datetime
import requests
import folium

# Third party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pydantic import BaseModel, ConfigDict
from pydantic import Field  # To add default values
import geopandas as gpd
from shapely.geometry import LineString
from geopy.distance import geodesic
from langchain_openai import ChatOpenAI


class FeatherFlight(BaseModel):
    """
    Class to get a better understanding of all our flights and drive the sustainable
    transition in our company.
    """

# Pydantic configuration

    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )  # allow pandas dataframes as types

    # Automatic and default attributes
    data_source: str = Field(
        "https://gitlab.com/adpro1/adpro2024/-/raw/main/Files/flight_data.zip?inline=false"
    )
    creation_date: datetime.date = datetime.date.today()

    # Dataframes declaration
    airlines: pd.DataFrame = None
    airplanes: pd.DataFrame = None
    airports: pd.DataFrame = None
    routes: pd.DataFrame = None

# Constructor

    def __init__(self, **kwargs) -> None:
        """We instantiate the class by downloading the data and adding it as attributes."""
        # 1. Call the parent class constructor for pydantic
        super().__init__(**kwargs)

        # 2. Download the data and download it to the downloads folder
        if not os.path.exists("./downloads"):
            os.makedirs("./downloads")
            try:
                req = requests.get(self.data_source, timeout=10)
                assert (
                    req.status_code == 200
                ), f"Failed to download file: {req.status_code}"
                with zipfile.ZipFile(io.BytesIO(req.content)) as zip_ref:
                    zip_ref.extractall("./downloads")
            except AssertionError as error:
                os.rmdir("./downloads")
                raise AssertionError(
                    f"{error} - (hint: check the data_source URL and your internet connection)"
                ) from error

        # 3. Add the data as an attribute to the class
        try:
            self.airlines = pd.read_csv("./downloads/airlines.csv")
            self.airplanes = pd.read_csv("./downloads/airplanes.csv")
            self.airports = pd.read_csv("./downloads/airports.csv")
            self.routes = pd.read_csv("./downloads/routes.csv")
            # Change the data type of the columns
            self.airports["Airport ID"] = self.airports["Airport ID"].astype(
                str)
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"{error} - (hint: check that the files are in the downloads folder)"
            ) from error

    def __str__(self) -> str:
        """Return a string representation of the class.

        Returns
        -------
        str
            Representation of the class.
        """
        return (
            f"FeatherFlight class created on the {self.creation_date}"
            + f" with data from '{self.data_source}'"
        )

# Method 0

    def calculate_distances_routes(self) -> None:
        """
        Calculate the distances between the source and destination airports for each route.

        Returns
        -------
        None
            A new column 'distance_km' is added to the routes dataframe. The coordinates of
            the source and destination airports are also added to the routes dataframe. 
        """
        # make sure to have the data in the same type
        self.airports["Airport ID"] = self.airports["Airport ID"].astype(str)
        self.routes["Source airport ID"] = self.routes["Source airport ID"].astype(
            str)
        self.routes["Destination airport ID"] = self.routes["Destination airport ID"].astype(
            str)

        # Add a column with the coordinates of the airport
        self.airports["Point"] = list(zip(
            self.airports.Latitude,
            self.airports.Longitude))

        # Add the coordinates of the source airport to the routes dataframe
        if "Point_source" not in self.routes.columns:
            self.routes = pd.merge(
                self.routes,
                self.airports[["Airport ID", "Point"]],
                how="left",
                left_on="Source airport ID",
                right_on="Airport ID",
            ).rename(columns={"Point": "Point_source"}).drop("Airport ID", axis=1)

        # Add the coordinates of the destination airport to the routes dataframe
        if "Point_destination" not in self.routes.columns:
            self.routes = pd.merge(
                self.routes,
                self.airports[["Airport ID", "Point"]],
                how="left",
                left_on="Destination airport ID",
                right_on="Airport ID",
            ).rename(columns={"Point": "Point_destination"}).drop("Airport ID", axis=1)

        # Calculate the distance between the source and destination airports
        # we use the geodesic function to account for the curvature of the earth
        # and overcome the limitations of the .distance method of GeoPandas
        self.routes["distance_km"] = self.routes.apply(
            lambda row: geodesic(
                row.Point_source, row.Point_destination).kilometers
            if pd.notnull(row.Point_source) and pd.notnull(row.Point_destination)
            else np.nan,
            axis=1
        )

# Method 1

    def country_airports(self, country: str) -> None:
        """
        Plot a map with the locations of the airports in a country.

        Parameters
        ----------

        country : str
            The name of the country.

        Returns
        -------
        None
            A map with the locations of the airports in the country is plotted.
        """

        filtered_airports = self.airports[self.airports["Country"] == country]
        if filtered_airports.empty:
            print(
                f"No airports found for {country}. " +
                "Please enter a valid country name.")
            return

        gdf_airports = gpd.GeoDataFrame(
            filtered_airports,
            geometry=gpd.points_from_xy(
                filtered_airports.Longitude, filtered_airports.Latitude
            ),
        )

        world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
        country_map = world[world["name"] == country]

        _, axis = plt.subplots(figsize=(10, 10))
        country_map.plot(ax=axis, color="lightgrey")
        gdf_airports.plot(ax=axis, marker="o", color="red", markersize=5)
        plt.title(f"Airports in {country}")
        plt.show()

# Method 2

    def distance_analysis(self) -> None:
        """
        Plot a histogram of the distribution of flight distances.

        Returns
        -------
        None
            A histogram of the distribution of flight distances is plotted.
        """
        # Compute the distances if not already done
        if "distance_km" not in self.routes.columns:
            self.calculate_distances_routes()

        # Plot the histogram
        plt.figure(figsize=(10, 6))
        self.routes["distance_km"].dropna().hist(bins=50, edgecolor="black")
        plt.title("Distribution of Flight Distances")
        plt.xlabel("Distance (km)")
        plt.ylabel("Number of Flights")
        plt.show()

# Method 3

    def plot_flights(self, airport: str, internal: bool = False) -> None:
        """Plot flights leaving the given airport.

        Parameters
        ----------
        airport : str
            The airport IATA code.
        internal : bool
            Whether to plot internal flights only. Defaults to False.

        Returns
        ------- 
        None
            A plot of the flights leaving the given airport is displayed.
        """

        airport_routes = self.routes[self.routes["Source airport"] == airport]

        if internal:
            # Filter routes within the same country
            country = self.airports[self.airports["IATA"]
                                    == airport]["Country"].iloc[0]
            airport_routes = airport_routes.merge(
                self.airports, left_on="Destination airport", right_on="IATA"
            )
            airport_routes = airport_routes[airport_routes["Country"] == country]

            # Convert airport_routes to a GeoDataFrame
            airport_routes = gpd.GeoDataFrame(
                airport_routes,
                geometry=gpd.points_from_xy(
                    airport_routes.Longitude, airport_routes.Latitude
                ),
            )

            # Read Natural Earth dataset file, and store geometries and attributes in a GeoDataFrame
            world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

        else:
            airport_routes = airport_routes.merge(
                self.airports, left_on="Destination airport", right_on="IATA"
            )
            # Convert airport_routes to a GeoDataFrame
            airport_routes = gpd.GeoDataFrame(
                airport_routes,
                geometry=gpd.points_from_xy(
                    airport_routes.Longitude, airport_routes.Latitude
                ),
            )

        departure_airport = self.airports[self.airports["IATA"] == airport]
        gdf_departure_airport = gpd.GeoDataFrame(
            departure_airport,
            geometry=gpd.points_from_xy(
                departure_airport.Longitude, departure_airport.Latitude
            ),
        )
        gdf_airport_routes = gpd.GeoDataFrame(
            airport_routes,
            geometry=gpd.points_from_xy(
                airport_routes.Longitude, airport_routes.Latitude
            ),
        )

        _, axis = plt.subplots(figsize=(10, 6))
        world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
        if internal:
            # Plot only the country
            world[world["name"] == country].plot(
                ax=axis, color="lightgray", aspect="equal", alpha=0.5
            )
        else:
            # Plot the whole world
            world.plot(ax=axis, color="lightgray", aspect="equal", alpha=0.5)

        # Plot airport routes and departure airport
        gdf_airport_routes.plot(ax=axis, color="blue",
                                label="Destination Airport")
        gdf_departure_airport.plot(
            ax=axis, color="red", label="Departure Airport")

        for _, row in airport_routes.iterrows():
            line = LineString(
                [
                    (
                        departure_airport["Longitude"].iloc[0],
                        departure_airport["Latitude"].iloc[0],
                    ),
                    (row["Longitude"], row["Latitude"]),
                ]
            )
            axis.plot(*line.xy, color="black", alpha=0.5)

            plt.text(
                row["Longitude"],
                row["Latitude"],
                row["Destination airport"],
                fontsize=8,
            )

        axis.set_title(f"Flights from {airport}")
        axis.set_xlabel("Longitude")
        axis.set_ylabel("Latitude")
        axis.legend()
        plt.grid(True)
        plt.show()

# Method 4

    def plot_top_airplane_models(self, countries: str = None, n_models: int = 5) -> None:
        """
        Plot the n most used airplane models by number of routes.

        Parameters
        ----------
        countries : str or list of str
            A country or a list of countries to filter the routes. Defaults to None.
        n_models : int
            Number of airplane models to plot. Defaults to 5.

        Returns
        -------
        None
            A bar plot of the n most used airplane models by number of routes is displayed.
        """
        # filter routes by country
        if countries is not None:
            if isinstance(countries, str):
                countries = [countries]
            # Merge routes with airports to get country information
            filtered_routes = self.routes.merge(
                self.airports, left_on="Source airport ID", right_on="IATA"
            )
            filtered_routes = filtered_routes[
                filtered_routes["Country"].isin(countries)
            ]
        else:
            filtered_routes = self.routes

        # Count the number of routes for each airplane model
        airplane_model_counts = filtered_routes["Equipment"].value_counts().head(
            n_models)

        plt.figure(figsize=(10, 6))
        airplane_model_counts.plot(kind="bar")
        plt.title(f"Top {n_models} Airplane Models by Number of Routes")
        plt.xlabel("Airplane Model")
        plt.ylabel("Number of Routes")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.show()

# Method 5

    def plot_flights_from_country(self, country: str, internal: bool = False, cutoff: float = 1000) -> None:
        """Plots all flights leaving from a given country to all other countries
        or only internal flights. Uses geopandas and matplotlib.

        Parameters
        -----------
        country : str
            Country name
        internal : bool, optional
            Plot internal flights only. Defaults to False.
        cutoff : float, optional
            Cutoff distance for short haul flights. Defaults to 1000.

        Returns
        --------
        None
            Plot of flights from the given country

        Raises
        -------
        AssertionError
            If the given country name is not in the airports dataset
        """

        try:
            assert (
                country in self.airports["Country"].unique()
            ), f"{country} is not in the airports dataset"

        except AssertionError as error:
            raise AssertionError(
                f"{error} - (hint: check the spelling of the country)"
            ) from error

        # filter airports to get the country's airports
        airports_ids = self.airports[
            self.airports["Country"] == country]["Airport ID"]

        # read the world map
        plot_map = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

        if internal:
            # filter routes to get the country's routes
            country_routes = self.routes[
                (self.routes["Source airport ID"].isin(airports_ids))
                & (self.routes["Destination airport ID"].isin(airports_ids))
            ]

            routes_codes = country_routes[
                ["Source airport ID", "Destination airport ID"]
            ]

            # reduce the map
            plot_map = plot_map[plot_map["name"] == country]

        else:
            country_routes = self.routes[
                self.routes["Source airport ID"].isin(airports_ids)
            ]

            routes_codes = country_routes[
                ["Source airport ID", "Destination airport ID"]
            ]

        # identify unique pairs of airports
        country_routes["Both Airports"] = country_routes.apply(
            lambda x: set([x["Source airport ID"], x["Destination airport ID"]]), axis=1
        )
        country_routes["Both Airports"] = country_routes["Both Airports"].apply(
            tuple)

        # short haul flights
        country_routes["Short Haul"] = country_routes["distance_km"] < cutoff
        country_routes["Short Haul"].value_counts(normalize=True)
        annot_before = country_routes.groupby("Short Haul").agg(
            {"distance_km": ["count", "sum"]})
        annot_before.columns = ["Number of Routes", "Total Distance"]
        annot_after = annot_before.copy()
        annot_before["Total Emmisions (kg)"] = annot_before["Total Distance"] * 0.151

        if False in annot_after.index:
            annot_after["Total Emmisions (kg)"] = [
                annot_after.loc[False, "Total Distance"] * 0.151,
                annot_after.loc[True, "Total Distance"] * 0.035,
            ]
        else:
            annot_after["Total Emmisions (kg)"] = [
                annot_after.loc[True, "Total Distance"] * 0.035
            ]

        # reduction percentage
        reduction_perc = annot_after["Total Emmisions (kg)"].sum(
        ) / annot_before["Total Emmisions (kg)"].sum() - 1

        # geometry
        geometry = country_routes.apply(
            lambda x: LineString(
                [x["Point_source"][::-1], x["Point_destination"][::-1]])
            if pd.notnull(x["Point_source"]) and pd.notnull(x["Point_destination"])
            else np.nan,
            axis=1
        )
        flights = gpd.GeoDataFrame(country_routes, geometry=geometry)

        # plot the map
        title = f"Flights from {country}"
        reduction = ''.join(
            [
                f"We could reduce the emissions by {reduction_perc:.2%} ",
                f"(from {annot_before['Total Emmisions (kg)'].sum():,.0f} to {annot_after['Total Emmisions (kg)'].sum():,.0f} kgCO2)",
                " by replacing short haul flights with rail."
            ]
        )

        title_html = f"""
            <h1 align="left" style="font-size:20px"><b>{title}</b></h1>
            <h2 align="left" style="font-size:16px"><b>{reduction}</b></h2>
        """

        # set the figure size of the map
        m = folium.Map(width=900, height=400, zoom_start=1)
        m.get_root().html.add_child(folium.Element(title_html))

        m = plot_map.explore(
            m=m,
            column="name",
            scheme="naturalbreaks",
            tooltip="name",
            name="Country",
            legend=False
        )

        flights[flights["Short Haul"]].explore(
            m=m,
            tooltip=[
                "Source airport",
                "Destination airport",
                "distance_km",
                "Short Haul",
            ],
            tooltip_kwds=dict(labels=True),
            style_kwds=dict(color="blue", weight=1, opacity=0.3),
            name="Short Haul Routes \n" +
            f"({annot_before.loc[True, 'Total Distance']:,.0f}km for {annot_before.loc[True, 'Number of Routes']:,} routes)"
        )

        if False in flights["Short Haul"].unique():
            flights[~flights["Short Haul"]].explore(
                m=m,
                tooltip=[
                    "Source airport",
                    "Destination airport",
                    "distance_km",
                    "Short Haul",
                ],
                tooltip_kwds=dict(labels=True),
                style_kwds=dict(color="red", weight=1, opacity=0.3),
                name="Long Haul Routes \n" +
                f"({annot_before.loc[False, 'Total Distance']:,.0f}km for {annot_before.loc[False, 'Number of Routes']:,} routes)"
            )

        folium.TileLayer("CartoDB positron", show=False).add_to(m)
        folium.LayerControl(collapsed=False).add_to(m)
        ploted_airports = self.airports[
            np.logical_or(
                self.airports["Airport ID"].isin(
                    routes_codes["Source airport ID"]),
                self.airports["Airport ID"].isin(
                    routes_codes["Destination airport ID"]),
            )
        ]

        sw = ploted_airports[['Latitude', 'Longitude']].min().values.tolist()
        ne = ploted_airports[['Latitude', 'Longitude']].max().values.tolist()

        m.fit_bounds([sw, ne])
        display(m)

    # Method 6
    def aircrafts(self) -> list:
        """Print the list containing all unique airplane models appearing in the airplanes dataset.

        Returns
        -------
        list 
            list of unique airplane models
        """
        print(
            f"Found {self.airplanes.Name.nunique()} airplane models",
            list(self.airplanes.Name.unique()),
        )

    # Method 7
    def aircraft_info(self, aircraft_name: str) -> str:
        """
        Get information about an aircraft model with the use of OpenAI 
        GPT-3.5-Turbo language model.

        Parameters
        ----------
        aircraft_name : str
            Name of the aircraft model

        Returns
        -------
        str
            Information about the aircraft model

        Raises
        ------
        ValueError
            If the aircraft name is not in the airplanes dataset
        """

        try:
            aircraft_name in self.airplanes.Name.unique()
        except ValueError as exc:
            raise ValueError(
                f"Invalid aircraft name. Please choose from the following list: {self.airplanes.Name.unique()}"
            ) from exc

        llm = ChatOpenAI(temperature=0, max_tokens=256)
        prompt = f'''Act as a search engine for providing information about aircraft models.
        After receiving the name of the aircraft model, you will provide a well structured list of its specification.
        Aircraft Model: {aircraft_name}'''
        response = llm.invoke(prompt)

        print(response.content)

        return response.content

    # Method 8
    def airport_info(self, airport_name_or_code: str) -> str:
        """Get information about an airport with the use of OpenAI GPT-3.5-Turbo language model.

        Parameters
        ----------
        airport_name_or_code : str
            Name or IATA code of the airport

        Returns
        -------
        str
            Information about the airport

        Raises
        ------
        ValueError
            If the airport name or code is not in the airports dataset
        """

        try:
            (
                airport_name_or_code not in self.airports.Name.unique()
                or airport_name_or_code not in self.airports.IATA.unique())
        except ValueError as exc:
            raise ValueError(
                f"Invalid airport name or code. Please choose from the following list: {self.airports.Name.unique()} or {self.airports.IATA.unique()}"
            ) from exc

        if (
            airport_name_or_code in self.airports.Name.unique()
            and airport_name_or_code not in self.airports.IATA.unique()
        ):
            airport_name = airport_name_or_code
            airport_code = self.airports[
                self.airports.Name == airport_name
            ].IATA.values[0]

        else:
            airport_code = airport_name_or_code
            airport_name = self.airports[
                self.airports.IATA == airport_code
            ].Name.values[0]

        llm = ChatOpenAI(temperature=0, max_tokens=256)
        prompt = f'''Act as a search engine for providing information about airports.
        After receiving the name or IATA code of the airport, you will provide a well structured list of its specification.
        Airport Name: {airport_name} \n Airport IATA Code: {airport_code}'''
        response = llm.invoke(prompt)

        print(response.content)

        return response.content


if __name__ == "__main__":
    flight = FeatherFlight()
    print(flight.airlines.head())
    print(flight.airplanes.head())
    print(flight.airports.head())
    print(flight.routes.head())
    print(flight.creation_date)
