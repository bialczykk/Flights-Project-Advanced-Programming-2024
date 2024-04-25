# Group_12



## FeatherFlight: Airline Operations Optimization Toolkit

FeatherFlight is a Python toolkit designed to assist data analysts in optimizing airline operations, reducing environmental impact, and enhancing route planning efficiency for client airlines. By leveraging flight data, FeatherFlight provides a comprehensive set of functionalities to analyze routes, aircraft usage, and airport connectivity.

This project is part of the course Advanced Programming for Data Science at NOVA School of Business and Economics. More information about the context and requirements of this project can be found under [project guidelines](https://gitlab.com/adpro1/adpro2024/-/blob/main/Project/Project_2324_Part_I.md?ref_type=heads).


## Visuals

Below visual representations generated by the methods developed in this project are presented. These visuals offer insightful aspects of airline operations that can be used for route optimization.

1. Map of the Airports in a Country:

This visual displays a map with the locations of airports within Portugal. It provides a geographical overview of airport distribution, aiding in understanding the country's air transportation infrastructure.

<img src="https://gitlab.com/SarahVranken/group_12/-/raw/main/images/image1.png" alt="Airports in Portugal" width="600">


2. Distribution of Flight Distances:

This visual illustrates the distribution of flight distances for all flights in the dataset. It helps in identifying patterns and trends in the lengths of flights, which can be valuable for route planning and optimization.

<img src="https://gitlab.com/SarahVranken/group_12/-/raw/main/images/image7.png" alt="Distribution of Flight Distances" width="600">


3. Flight Routes from an Airport:

This visuals represents the flight routes originating from Lisbon Airport. In the first visual, all flights from the airport are displayed. In the second visual, only flights with destinations within the same country as the airport are plotted. 

<img src="https://gitlab.com/SarahVranken/group_12/-/raw/main/images/image2.png" alt="Flights from Lisbon" width="600">

<img src="https://gitlab.com/SarahVranken/group_12/-/raw/main/images/image3.png" alt="Internal Flights from Lisbon" width="600">

4. Most Used Airplane Models by Number of Routes:

This visual presents the top 10 most used airplane models based on the number of routes they serve.

<img src="https://gitlab.com/SarahVranken/group_12/-/raw/main/images/image4.png" alt="Most Used Airplane Models by Number of Routes" width="600">


5. Flight Routes from a Country with Emmission Reduction Potential:

These visuals showcase flight routes departing from Portugal. In the first visual, all flights from Portugal are displayed. In the second visual, only flights with destinations within Portugal itself are plotted. Furthermore, flights are categorized into short-haul and long-haul flights. This provides insights into the distribution of flight distances, crucial for strategic planning. In addition to providing an overview of flight operations, the potential decrease in flight emissions by replacing short-haul flights with rail services is highlighted in the plot. 

<img src="https://gitlab.com/SarahVranken/group_12/-/raw/main/images/image8.png" alt="Flights from Portugal" width="600">

<img src="https://gitlab.com/SarahVranken/group_12/-/raw/main/images/image9.png" alt="Internal Flights from Portugal" width="600">

For more details, please check the [Showcase Notebook](https://gitlab.com/SarahVranken/group_12/-/blob/main/showcase_notebook.ipynb?ref_type=heads).


## Installation

To install FeatherFlight, follow these steps:

1. Clone the repository to your local machine:

```
git clone git@gitlab.com:SarahVranken/group_12.git
```
2. Navigate to the project directory:

```
cd group_12
```

3. Set up your environment

3.1. With `poetry`
This repository is managed with [poetry](https://python-poetry.org/) which makes it easy to manage dependencies and virtual environments. To create a virtual environment and install the dependencies, run the following commands:

```
# Create and activate the virtual environment
$ python -m venv .venv
$ .venv\Scripts\activate

# Install poetry and the dependencies
$ pip install poetry
$ poetry install
```

3.2. With `pip`

If you prefer to use `pip`, you can install the dependencies with the following command:

```
pip install -r requirements.pip
```

3.3. With conda
If Conda is the prefered environment management tool, the environment for this project can be initiatied with the following command:
```
conda env create -f environment.yml
```


4. Run the tests to ensure everything is working correctly:

To run the tests, navigate to the `tests` directory (`cd tests`) and run the following command:

```
pytest test_distances.py
pytest test_flights_countries.py
``` 

## Usage

FeatherFlight provides a variety of methods to analyze and visualize airline operations. Below are some examples of usage:

```
from featherflight import FeatherFlight

# Initialize FeatherFlight instance
ff = FeatherFlight()

# Analyze flight distances
ff.distance_analysis()

# Plot airports within a specific country
ff.country_airports('Portugal')

# Plot departing flights from a specific airport
ff.plot_flights('LIS')

# Plot top airplane models used
ff.plot_top_airplane_models()

# Plot flights from a specific country
ff.plot_flights_from_country('Portugal')
```

For more detailed usage, please check the [Showcase Notebook](https://gitlab.com/SarahVranken/group_12/-/blob/main/showcase_notebook.ipynb?ref_type=heads)


## Support

For help and support, please [create an issue](https://gitlab.com/SarahVranken/group_12/-/issues) on the GitHub repository.


## Roadmap

Implement additional features for route optimization:
- Algorithmic Route Optimization:

    Develop algorithms to automatically optimize airline routes based on vaious criteria, such as fuel efficiency, flight duration, and passenger demand.
- Dynamic Route Planning:
        
    Implement dynamic route planning capabilities that allow airlines to adapt routes in real-time based on changing factors, such as weather conditions, and airspace restrictions.
- Route Performance Analysis:
        
    Develop tools to analyze the performance of existing routes and identify opportunities for optimization. This could involve analyzing historical flight data to identify trends, patterns, and inefficiencies in route performance, and using this information to inform route optimization decisions.


## Contributing

To contribute to FeatherFlight, follow these steps:

1. Fork the repository
2. Create a new branch (git checkout -b feature/new-feature)
3. Make your changes
4. Commit your changes (git commit -am 'Add new feature')
5. Push to the branch (git push origin feature/new-feature)
6. Create a new Pull Request


## Authors and acknowledgment

- Sarah Vranken: <a href="mailto:63283@novasbe.pt">63283@novasbe.pt</a>
- Mathieu Julien R. Demarets: <a href="mailto:63282@novasbe.pt">63282@novasbe.pt</a>
- Andrea Piredda: <a href="mailto:61801@novasbe.pt">61801@novasbe.pt</a>
- Kuba Maciej Bialczyk: <a href="mailto:61678@novasbe.pt">61678@novasbe.pt</a>


## License

This project is licensed under the GNU GENERAL PUBLIC LICENSE. See the [LICENSE](LICENSE) file for details.


## Structure of the project

FeatherFlight follows a structured project layout to ensure clarity and organization:

- Showcase Notebook: A notebook showcasing the functionality of FeatherFlight.
- scripts/: Directory containing all Python scripts for the project.
- tests/: Directory for storing the tests data files.
- docs/: Directory for storing the files to set up Sphinx documentation.
- images/: Directory for storing the images displayed in this README file.
- .gitignore: File specifying which files and directories to ignore in version control.
- LICENSE: License file specifying the terms and conditions for using the project
- poetry.lock and pyproject.toml: to manage dependencies and virtual environment with poetry.
- requirements.pip: File specifying the dependencies for the project to manage the virtual environment with pip.