"""
Utilities for loading Montreal public transit scheduling instances.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class Trip:
    """A scheduled passenger trip from voyages.txt."""

    trip_id: str
    departure_node: str
    departure_time: int
    arrival_node: str
    arrival_time: int
    line_id: str


@dataclass(frozen=True)
class ProblemInstance:
    """Parsed input files for one scheduling instance."""

    instance_path: str
    depots: Dict[str, int]
    recharge_stations: Dict[str, int]
    trips: List[Trip]
    deadhead_times: Dict[Tuple[str, str], int]
    operation_start_min: int
    operation_end_min: int
    max_deadhead_time: int
    line_ids: List[str]


def _split_semicolon_line(raw: str) -> List[str]:
    return [part.strip() for part in raw.strip().split(";") if part.strip()]


def _line_sort_key(line_id: str) -> Tuple[int, str]:
    try:
        return (0, f"{int(line_id):09d}")
    except ValueError:
        return (1, line_id)


def load_problem_instance(
    dataset_root: str = "dataArticleJuliette",
    subset: str = "A",
    split: str = "Training",
    instance: str = "Network9a_22_0",
) -> ProblemInstance:
    """
    Parse depots, recharge, trips, and deadhead travel times for one instance.
    """

    base_path = Path(dataset_root) / subset / split / instance
    if not base_path.exists():
        raise FileNotFoundError(f"Instance path not found: {base_path}")

    depots = _parse_depots(base_path / "depots.txt")
    recharge_stations = _parse_recharge(base_path / "recharge.txt")
    trips = _parse_trips(base_path / "voyages.txt")
    deadhead_times = _parse_deadhead_times(base_path / "hlp.txt")

    if not trips:
        raise ValueError(f"No trips found in {base_path / 'voyages.txt'}")

    operation_start_min = min(trip.departure_time for trip in trips)
    operation_end_min = max(trip.arrival_time for trip in trips)
    max_deadhead_time = max(deadhead_times.values()) if deadhead_times else 1
    line_ids = sorted({trip.line_id for trip in trips}, key=_line_sort_key)

    return ProblemInstance(
        instance_path=str(base_path),
        depots=depots,
        recharge_stations=recharge_stations,
        trips=trips,
        deadhead_times=deadhead_times,
        operation_start_min=operation_start_min,
        operation_end_min=operation_end_min,
        max_deadhead_time=max_deadhead_time,
        line_ids=line_ids,
    )


def _parse_depots(path: Path) -> Dict[str, int]:
    depots: Dict[str, int] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        parts = _split_semicolon_line(raw)
        if len(parts) < 2:
            continue
        depots[parts[0]] = int(parts[1])
    if not depots:
        raise ValueError(f"No depots parsed from {path}")
    return depots


def _parse_recharge(path: Path) -> Dict[str, int]:
    stations: Dict[str, int] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        parts = _split_semicolon_line(raw)
        if len(parts) < 2:
            continue
        stations[parts[0]] = int(parts[1])
    return stations


def _parse_trips(path: Path) -> List[Trip]:
    trips: List[Trip] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        parts = _split_semicolon_line(raw)
        if len(parts) < 6:
            continue
        trips.append(
            Trip(
                trip_id=parts[0],
                departure_node=parts[1],
                departure_time=int(parts[2]),
                arrival_node=parts[3],
                arrival_time=int(parts[4]),
                line_id=parts[5],
            )
        )
    trips.sort(key=lambda t: (t.departure_time, _line_sort_key(t.line_id), t.trip_id))
    return trips


def _parse_deadhead_times(path: Path) -> Dict[Tuple[str, str], int]:
    deadhead_times: Dict[Tuple[str, str], int] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        parts = _split_semicolon_line(raw)
        if len(parts) < 3:
            continue
        node_from = parts[0]
        node_to = parts[1]
        travel_time = int(parts[2])
        key = (node_from, node_to)
        if key not in deadhead_times:
            deadhead_times[key] = travel_time
        else:
            deadhead_times[key] = min(deadhead_times[key], travel_time)
    return deadhead_times
