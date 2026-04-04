import math
import os
from typing import Dict, List

import numpy as np

from config import (
    ACTION_DIM,
    BUS_LINES,
    COORDINATES,
    DATASET_INSTANCE,
    DATASET_ROOT,
    DATASET_SPLIT,
    DATASET_SUBSET,
    DATA_SOURCE,
    DEPOT,
    INITIAL_NUM_BUSES,
    MAX_DISTANCE,
    OPERATION_END_MIN,
    OPERATION_START_MIN,
    W_CHAIN,
    W_DEADHEAD,
    W_DEMAND_PENALTY,
    W_FINAL,
    W_REST_REWARD,
    W_UNAVAILABILITY,
    W_UNUSED_PENALTY,
)
from dataset_loader import load_problem_instance


class BusSchedulingEnv:
    """
    Bus scheduling environment that supports:
    1) synthetic data from config.py, and
    2) Juliette dataset instances from dataArticleJuliette.
    """

    def __init__(
        self,
        data_source: str = None,
        dataset_root: str = None,
        dataset_subset: str = None,
        dataset_split: str = None,
        dataset_instance: str = None,
    ):
        self.data_source = data_source or DATA_SOURCE
        self.dataset_root = dataset_root or DATASET_ROOT
        self.dataset_subset = dataset_subset or DATASET_SUBSET
        self.dataset_split = dataset_split or DATASET_SPLIT
        self.dataset_instance = dataset_instance or DATASET_INSTANCE

        self.instance_label = "synthetic"
        self.recharge_stations: Dict[str, int] = {}
        self.depot_nodes: List[str] = [DEPOT]
        self.deadhead_times: Dict[tuple, int] = {}

        if self.data_source == "juliette":
            self._init_juliette_problem()
        else:
            self._init_synthetic_problem()

        self.num_events = len(self.timetable)
        self.current_index = 0

        global MAX_EPISODE_STEPS
        MAX_EPISODE_STEPS = self.num_events

        self.bus_status = self._create_initial_bus_status()
        self.schedule = {bus_id: [] for bus_id in range(self.num_buses)}

        self.observation_space_dim = 2 + self.num_buses
        self.action_space_dim = self.num_buses

        # Backward compatibility for older code that imports ACTION_DIM from config.
        if ACTION_DIM != self.action_space_dim and self.data_source == "synthetic":
            self.action_space_dim = ACTION_DIM

    def _init_synthetic_problem(self):
        self.operation_start_min = OPERATION_START_MIN
        self.operation_end_min = OPERATION_END_MIN
        self.t_range = max(1, self.operation_end_min - self.operation_start_min)
        self.num_buses = INITIAL_NUM_BUSES
        self.depot_nodes = [DEPOT]
        self.initial_bus_locations = [DEPOT for _ in range(self.num_buses)]
        self.timetable = self._generate_synthetic_timetable()
        line_ids = sorted(BUS_LINES.keys())
        self.line_to_index = {line_id: idx + 1 for idx, line_id in enumerate(line_ids)}
        self.max_deadhead_time = max(1.0, MAX_DISTANCE)

    def _init_juliette_problem(self):
        problem = load_problem_instance(
            dataset_root=self.dataset_root,
            subset=self.dataset_subset,
            split=self.dataset_split,
            instance=self.dataset_instance,
        )
        self.instance_label = problem.instance_path
        self.recharge_stations = problem.recharge_stations
        self.depot_nodes = sorted(problem.depots.keys())
        self.operation_start_min = problem.operation_start_min
        self.operation_end_min = problem.operation_end_min
        self.t_range = max(1, self.operation_end_min - self.operation_start_min)
        self.deadhead_times = problem.deadhead_times
        self.max_deadhead_time = max(1, problem.max_deadhead_time)

        self.initial_bus_locations = []
        for depot_node, count in sorted(problem.depots.items()):
            self.initial_bus_locations.extend([depot_node] * count)
        self.num_buses = len(self.initial_bus_locations)

        self.timetable = []
        for trip in problem.trips:
            self.timetable.append(
                {
                    "trip_id": trip.trip_id,
                    "time": trip.departure_time,
                    "departure_time": trip.departure_time,
                    "arrival_time": trip.arrival_time,
                    "line_id": trip.line_id,
                    "departure_node": trip.departure_node,
                    "arrival_node": trip.arrival_node,
                    "trip_time": max(0, trip.arrival_time - trip.departure_time),
                    "rest_time": 0,
                }
            )

        self.line_to_index = {line_id: idx + 1 for idx, line_id in enumerate(problem.line_ids)}

    def _generate_synthetic_timetable(self):
        events = []
        for line_id, info in BUS_LINES.items():
            interval = info["interval"]
            departure = OPERATION_START_MIN
            while departure <= OPERATION_END_MIN:
                trip_time = info["trip_time"]
                rest_time = info["rest_time"]
                events.append(
                    {
                        "trip_id": f"SYN-{line_id}-{departure}",
                        "time": departure,
                        "departure_time": departure,
                        "arrival_time": departure + trip_time,
                        "line_id": line_id,
                        "departure_node": info["terminal"],
                        "arrival_node": info["terminal"],
                        "trip_time": trip_time,
                        "rest_time": rest_time,
                    }
                )
                departure += interval
        events.sort(key=lambda e: (e["departure_time"], str(e["line_id"])))
        return events

    def _create_initial_bus_status(self):
        status = {}
        for bus_id, start_node in enumerate(self.initial_bus_locations):
            status[bus_id] = {
                "location": start_node,
                "next_available_time": self.operation_start_min,
                "used": False,
            }
        return status

    def _get_deadhead_time(self, current_location, required_location):
        if current_location == required_location:
            return 0.0

        if self.data_source == "juliette":
            return float(self.deadhead_times.get((current_location, required_location), self.max_deadhead_time))

        x1, y1 = COORDINATES[current_location]
        x2, y2 = COORDINATES[required_location]
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def _normalized_deadhead_cost(self, current_location, required_location):
        deadhead_time = self._get_deadhead_time(current_location, required_location)
        return deadhead_time / max(1e-6, self.max_deadhead_time)

    def get_valid_actions(self):
        if self.current_index >= self.num_events:
            return []

        event = self.timetable[self.current_index]
        event_time = event["departure_time"]
        required_node = event["departure_node"]

        valid_actions = []
        for bus_id in range(self.num_buses):
            bus_info = self.bus_status[bus_id]
            ready_time = bus_info["next_available_time"] + self._get_deadhead_time(
                bus_info["location"], required_node
            )
            if ready_time <= event_time:
                valid_actions.append(bus_id)

        # Keep agents robust when no feasible bus is found.
        if not valid_actions:
            return list(range(self.num_buses))
        return valid_actions

    def reset(self):
        self.current_index = 0
        self.bus_status = self._create_initial_bus_status()
        self.schedule = {bus_id: [] for bus_id in range(self.num_buses)}
        return self.get_state()

    def get_state(self):
        if self.current_index >= self.num_events:
            return np.zeros(self.observation_space_dim, dtype=np.float32)

        event = self.timetable[self.current_index]
        event_time = event["departure_time"]
        required_node = event["departure_node"]

        state = np.zeros(self.observation_space_dim, dtype=np.float32)
        state[0] = (event_time - self.operation_start_min) / self.t_range

        line_count = max(1, len(self.line_to_index))
        line_idx = self.line_to_index.get(event["line_id"], 1)
        state[1] = (line_idx - 1) / max(1, line_count - 1)

        for bus_id in range(self.num_buses):
            bus_info = self.bus_status[bus_id]
            ready_time = bus_info["next_available_time"] + self._get_deadhead_time(
                bus_info["location"], required_node
            )
            state[2 + bus_id] = (event_time - ready_time) / self.t_range

        return state

    def step(self, action):
        if self.current_index >= self.num_events:
            return self.get_state(), 0.0, True, {}

        if action < 0 or action >= self.num_buses:
            raise ValueError(f"Invalid action {action}. Valid range is [0, {self.num_buses - 1}]")

        event = self.timetable[self.current_index]
        event_time = event["departure_time"]
        required_node = event["departure_node"]

        bus_info = self.bus_status[action]
        current_location = bus_info["location"]
        deadhead_time = self._get_deadhead_time(current_location, required_node)
        deadhead_cost = self._normalized_deadhead_cost(current_location, required_node)
        ready_time = bus_info["next_available_time"] + deadhead_time

        penalty_unavail = W_UNAVAILABILITY if ready_time > event_time else 0.0

        rn = 0.0
        if (not bus_info["used"]) and any(
            self.bus_status[b]["used"]
            and self.bus_status[b]["location"] in self.depot_nodes
            and (
                self.bus_status[b]["next_available_time"]
                + self._get_deadhead_time(self.bus_status[b]["location"], required_node)
                <= event_time
            )
            for b in self.bus_status
        ):
            rn = 1.0

        chain_bonus = 0.0
        if self.schedule[action]:
            last_event = self.schedule[action][-1]
            if last_event["line_id"] == event["line_id"]:
                chain_bonus = W_CHAIN

        rk = 1.0 if (bus_info["used"] and ready_time <= event_time) else 0.0
        ru = 0.0

        step_reward = (
            -(W_UNUSED_PENALTY * rn + W_DEADHEAD * deadhead_cost + penalty_unavail)
            + W_REST_REWARD * rk
            + chain_bonus
            - W_DEMAND_PENALTY * ru
        )

        self.schedule[action].append(event)
        self.bus_status[action]["used"] = True
        self.bus_status[action]["location"] = event["arrival_node"]
        self.bus_status[action]["next_available_time"] = event["arrival_time"] + event.get("rest_time", 0)

        self.current_index += 1
        done = self.current_index >= self.num_events
        next_state = self.get_state()

        if done:
            num_used = sum(1 for bus_id in self.bus_status if self.bus_status[bus_id]["used"])
            step_reward += -W_FINAL * num_used

        info = {
            "event": event,
            "deadhead_time": deadhead_time,
            "deadhead_cost": deadhead_cost,
            "rn": rn,
            "penalty_unavail": penalty_unavail,
            "rk": rk,
            "chain_bonus": chain_bonus,
        }
        return next_state, step_reward, done, info

    def get_total_buses_used(self):
        return sum(1 for bus_id in self.bus_status if self.bus_status[bus_id]["used"])

    def print_and_save(self, text, file):
        print(text, end="")
        file.write(text)

    def print_problem(self, file_path="data/results.txt"):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as handle:
            self.print_and_save("=== Problem Definition ===\n", handle)

            start_hr, start_min = divmod(self.operation_start_min, 60)
            end_hr, end_min = divmod(self.operation_end_min, 60)
            self.print_and_save(
                f"Operation Period: {start_hr:02d}:{start_min:02d} to {end_hr:02d}:{end_min:02d}\n\n",
                handle,
            )

            if self.data_source == "juliette":
                self.print_and_save(f"Data Source: juliette ({self.instance_label})\n", handle)
                self.print_and_save(f"Depots: {len(self.depot_nodes)}\n", handle)
                for depot_node in self.depot_nodes:
                    depot_count = self.initial_bus_locations.count(depot_node)
                    self.print_and_save(f"  {depot_node}: {depot_count} buses\n", handle)

                self.print_and_save(f"Recharge Stations: {len(self.recharge_stations)}\n", handle)
                for node, chargers in sorted(self.recharge_stations.items()):
                    self.print_and_save(f"  {node}: {chargers} chargers\n", handle)

                self.print_and_save(f"Unique Lines: {len(self.line_to_index)}\n\n", handle)
                self.print_and_save("Timetable (Voyages):\n", handle)
                for event in self.timetable:
                    dep_hr, dep_min = divmod(event["departure_time"], 60)
                    arr_hr, arr_min = divmod(event["arrival_time"], 60)
                    self.print_and_save(
                        f"  {event['trip_id']}: {dep_hr:02d}:{dep_min:02d} {event['departure_node']} -> "
                        f"{arr_hr:02d}:{arr_min:02d} {event['arrival_node']} (Line: {event['line_id']})\n",
                        handle,
                    )
            else:
                self.print_and_save("Bus Lines:\n", handle)
                for line_id, info in BUS_LINES.items():
                    self.print_and_save(
                        f"  Bus Line {line_id}: {info['name']} - Loop at {info['terminal']}, "
                        f"Interval: {info['interval']} min (Trip: {info['trip_time']} + Rest: {info['rest_time']})\n",
                        handle,
                    )
                self.print_and_save("\nTimetable (Departure Events):\n", handle)
                for event in self.timetable:
                    hr, minute = divmod(event["departure_time"], 60)
                    self.print_and_save(
                        f"  {hr:02d}:{minute:02d} - Bus Line {event['line_id']} "
                        f"(Terminal: {event['departure_node']})\n",
                        handle,
                    )

            self.print_and_save("==========================\n\n", handle)

    def print_solution(self, file_path="data/results.txt"):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "a", encoding="utf-8") as handle:
            self.print_and_save("=== Final Bus Schedules (Solution) ===\n", handle)
            for bus_id, events in self.schedule.items():
                if events:
                    self.print_and_save(f"Bus {bus_id} schedule:\n", handle)
                    for event in events:
                        dep_hr, dep_min = divmod(event["departure_time"], 60)
                        arr_hr, arr_min = divmod(event["arrival_time"], 60)
                        self.print_and_save(
                            f"  {event['trip_id']} | {dep_hr:02d}:{dep_min:02d} {event['departure_node']} -> "
                            f"{arr_hr:02d}:{arr_min:02d} {event['arrival_node']} "
                            f"(Line: {event['line_id']})\n",
                            handle,
                        )
                else:
                    self.print_and_save(f"Bus {bus_id} was not used.\n", handle)
            self.print_and_save("========================================\n", handle)
