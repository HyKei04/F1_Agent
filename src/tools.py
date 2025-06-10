import base64
from typing import List, Literal
import fastf1
import fastf1.plotting
import matplotlib.pyplot as plt
import seaborn as sns
from smolagents import Tool, tool
import pandas as pd
from langchain_core.vectorstores import VectorStore
from litellm import completion


@tool
def plot_fastest_lap_telemetry(drivers: List[str], year: int, gp: str, session_type: str,
                               variables: List[Literal["Speed", "RPM", "Gear", "Throttle", "Brake", "DRS"]] =
                               ["Speed"], output_path: str = "output.png") -> str:
    """
    Builds a graph comparing the telemetry of the fastest laps for multiple
    Formula 1 drivers in a session.

    Args:
        drivers (List[str]): List of three-letter driver abbreviations (e.g., ["VER", "HAM", "LEC"]).
        year (int): Year of the Grand Prix.
        gp (str): Official name of the Grand Prix (e.g., "Spanish Grand Prix").
        session_type (str): Session type: "Q", "R", "FP1", etc.
        variables (List[str]): List of telemetry variables to plot (e.g., ["Speed", "Throttle"]).
        output_path (str): File path where the output image will be saved.

    Returns:
        str: Path to the saved plot image.
    """

    if("Gear" in variables):
        variables.remove("Gear")
        variables.append("nGear")

    unit_labels = {
        "Speed": "Speed (km/h)",
        "Throttle": "Throttle (%)",
        "Brake": "Brake",
        "RPM": "RPM",
        "nGear": "Gear",
        "DRS": "DRS (0/1)"
    }

    fastf1.plotting.setup_mpl(mpl_timedelta_support=True, misc_mpl_mods=False, color_scheme='fastf1')

    session = fastf1.get_session(year, gp, session_type)
    session.load()

    circuit_info = session.get_circuit_info()

    fig, axs = plt.subplots(len(variables), 1, figsize=(10, 5 * len(variables)), sharex=True)
    if len(variables) == 1:
        axs = [axs]

    legend_entries = []

    for driver in drivers:
        lap = session.laps.pick_drivers(driver).pick_fastest()
        telemetry = lap.get_car_data().add_distance()
        color = fastf1.plotting.get_team_color(lap['Team'], session=session)

        lap_time = lap['LapTime']
        lap_time_str = str(lap_time)[-11:-3] if lap_time else "N/A"  # format: mm:ss.xxx
        driver_label = f"{driver} ({lap_time_str})"
        legend_entries.append((driver_label, color))

        for i, var in enumerate(variables):
            if var not in telemetry.columns:
                raise ValueError(f"Variable '{var}' not found in telemetry data.")
            axs[i].plot(telemetry['Distance'], telemetry[var], label=driver_label, color=color)

            ylabel = unit_labels.get(var, var)  # fallback to var name if not found
            axs[i].set_ylabel(ylabel)
            ymin, ymax = axs[i].get_ylim()
            axs[i].vlines(x=circuit_info.corners['Distance'], ymin=ymin, ymax=ymax,
                          linestyles='dotted', colors='grey')
            axs[i].grid(True, axis='y', linestyle='-', alpha=0.3)
            axs[i].set_xlim(left=0)

            if var == "Brake":
                axs[i].set_yticks([0, 1])
                axs[i].set_yticklabels(["OFF", "ON"])

            if i < len(variables) - 1:
                axs[i].tick_params(labelbottom=False)

    axs[-1].set_xlabel("Distance (m)")

    # Create single legend
    handles = [plt.Line2D([], [], color=color, label=label) for label, color in legend_entries]
    axs[0].legend(handles=handles, loc="upper right", framealpha=0.2)

    plt.suptitle(f"{session.event['EventName']} {year} - {session_type.upper()} Telemetry Comparison")
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(output_path)
    plt.close(fig)

    return output_path


@tool
def plot_race_pace(year: int, gp: str, drivers: List[str] = None, output_path: str = "output.png") -> str:
    """
    Makes a boxplot graph comparing the race pace of the drivers given.Lang
    If no drivers are given by parameter, all drivers are included.

    Args:
        year (int): Year of the Grand Prix.
        gp (str): Official name of the Grand Prix (e.g., "Spanish Grand Prix").
        drivers (List[str]): List of three-letter driver abbreviations (e.g., ["VER", "HAM", "LEC"]).
        output_path (str): File path where the output image will be saved.

    Returns:
        str: Path to the saved plot image.
    """

    fastf1.plotting.setup_mpl(mpl_timedelta_support=False, misc_mpl_mods=False, color_scheme='fastf1')

    session = fastf1.get_session(year, gp, "Race")
    session.load()
    laps = session.laps.pick_quicklaps(threshold=1.10).pick_wo_box().pick_accurate()

    if drivers:
        laps = laps.pick_drivers(drivers)

    transformed_laps = laps.copy()
    transformed_laps.loc[:, "LapTime (s)"] = laps["LapTime"].dt.total_seconds()

    # Calculate average lap times per driver
    mean_times = transformed_laps.groupby("Driver")["LapTime (s)"].mean().sort_values()
    first_driver_mean = mean_times.iloc[0]
    gap_to_first = mean_times - first_driver_mean

    # Format driver labels with mean + gap
    formatted_labels = [
        f"{driver}\n{mean_times[driver]:.2f}s\n+{gap_to_first[driver]:.2f}s" if gap_to_first[driver] != 0
        else f"{driver}\n{mean_times[driver]:.2f}s\nLeader"
        for driver in mean_times.index
    ]

    transformed_laps["Driver"] = pd.Categorical(transformed_laps["Driver"], categories=mean_times.index, ordered=True)
    driver_palette = {
        driver: fastf1.plotting.get_team_color(transformed_laps.pick_drivers(driver).iloc[0]["Team"], session=session)
        for driver in mean_times.index
    }

    fig, ax = plt.subplots(figsize=(15, 10))
    sns.boxplot(
        data=transformed_laps,
        x="Driver",
        y="LapTime (s)",
        hue="Driver",
        order=mean_times.index,
        palette=driver_palette,
        whiskerprops=dict(color="white"),
        boxprops=dict(edgecolor="white"),
        medianprops=dict(color="grey"),
        capprops=dict(color="white"),
    )

    max_lap_time = transformed_laps["LapTime (s)"].max()
    ax.set_ylim(top=max_lap_time + 1)
    ax.set_xticks(range(len(formatted_labels)))
    ax.set_xticklabels(formatted_labels, fontsize=10)

    plt.title(f"{session.event['EventName']} {year} - Race")
    plt.grid(visible=True, axis='y', linestyle='-', alpha=0.3)
    ax.set(xlabel=None)
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    return output_path

@tool
def plot_race_strategy(year: int, gp: str, drivers: List[str] = None, output_path: str = "output.png") -> str:
    """
    Builds a graph comparing the race strategy of the given drivers. If no drivers are given it compares all of them

    Args:
        year (int): Year of the Grand Prix.
        gp (str): Official name of the Grand Prix (e.g., "Spanish Grand Prix").
        drivers (List[str]): List of three-letter driver abbreviations (e.g., ["VER", "HAM", "LEC"]).
        output_path (str): File path where the output image will be saved.

    Returns:
        str: Path to the saved plot image.
    """
    fastf1.plotting.setup_mpl(mpl_timedelta_support=True, misc_mpl_mods=False, color_scheme='fastf1')

    session = fastf1.get_session(year, gp, "R")
    session.load()
    laps = session.laps

    if drivers is None or len(drivers) == 0:
        drivers = session.drivers
        drivers = [session.get_driver(driver)["Abbreviation"] for driver in drivers]

    stints = laps[["Driver", "Stint", "Compound", "LapNumber"]]
    stints = stints.groupby(["Driver", "Stint", "Compound"])
    stints = stints.count().reset_index()

    stints = stints.rename(columns={"LapNumber": "StintLength"})
    fig, ax = plt.subplots(figsize=(5, 10))

    for driver in drivers:
        driver_stints = stints.loc[stints["Driver"] == driver]
        previous_stint_end = 0
        for idx, row in driver_stints.iterrows():
            compound_color = fastf1.plotting.get_compound_color(row["Compound"],
                                                                session=session)
            plt.barh(
                y=driver,
                width=row["StintLength"],
                left=previous_stint_end,
                color=compound_color,
                edgecolor="black",
                fill=True
            )
            previous_stint_end += row["StintLength"]

    plt.title(f"{session.event['EventName']} {year} - Race Strategies", fontsize=12)
    plt.xlabel("Lap Number")
    plt.grid(False)

    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    return output_path

@tool
def analize_graph(graph_url: str) -> str:
    """
    Performs a detailed analysis of the given F1-related graph

    Args:
        graph_url (str): URL of the F1-related graph

    Returns:
        str: Analysis of the graph
    """
    with open(graph_url, "rb") as image_file:
        image_bytes = image_file.read()
        base64_image = base64.b64encode(image_bytes).decode('utf-8')

    response = completion(model="openrouter/openai/gpt-4o-mini",
                          messages=[{"role": "user",
                                     "content": [
                                         {
                                             "type": "text",
                                             "text": "Analize this image containing a F1-related graph"
                                         },
                                         {
                                             "type": "image_url",
                                             "image_url": {
                                                 "url": f"data:image/png;base64,{base64_image}"
                                             }
                                         }
                                     ]}],
                          max_tokens=1000)
    return response.choices[0].message.content


class RetrieverTool(Tool):
    name = "f1_info_retriever"
    description = ("Uses lexical search to retrieve information related to Formula 1, such as technical changes made by"
                   "teams or penalties applied, that could be most relevant to answer your query.")
    inputs = {
        "query": {
            "type": "string",
            "description": "The race you want information about including which type of information (technical changes"
                           "or penalties)",
        }
    }
    output_type = "string"

    def __init__(self, vectordb: VectorStore, **kwargs):
        super().__init__(**kwargs)
        self.vectordb = vectordb

    def set_description_for_race(self, race: str):
        self.name = f"f1_{race}_grand_prix_info_retriever"
        self.description = f"Retrieves F1-related information (penalties or technical changes) for {race} Grand Prix."

    def set_description_for_regulations(self):
        self.name = "f1_sporting_regulations_info_retriever"
        self.description = "Retrieves F1-related information about the sporting regulations."

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "Your search query must be a string"

        docs = self.vectordb.similarity_search(
            query,
            k=10,
        )
        return "Retrieved documents:\n\n" + "\n===Document===\n".join([doc.page_content for doc in docs])
