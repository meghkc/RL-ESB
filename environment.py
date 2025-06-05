# environment.py
import numpy as np
import math
import os
from config import (OPERATION_START_MIN, OPERATION_END_MIN, T_RANGE, BUS_LINES, DEPOT, INITIAL_NUM_BUSES,
                    STATE_DIM, COORDINATES, MAX_DISTANCE, W_DEADHEAD, W_UNUSED_PENALTY, 
                    W_REST_REWARD, W_UNAVAILABILITY, W_DEMAND_PENALTY, W_CHAIN, W_FINAL)

class BusSchedulingEnv:
    """
    Bus Scheduling Environment with time dynamics and chaining bonus.
    - Generates a timetable from bus lines.
    - All buses start at the depot; each bus has a next_available_time.
    - The state vector includes: normalized current event time, normalized bus line,
      and for each bus a continuous value = (current event time - next_available_time) / T_RANGE.
    - The reward function includes:
         • Deadhead cost if bus is not at required terminal,
         • Penalty for using an unused bus when a used bus is available,
         • Penalty if a bus is not available (negative availability),
         • Bonus for reusing a bus on the same bus line consecutively.
    - At the end, a final penalty proportional to the number of buses used is applied.
    """
    def __init__(self):
        self.timetable = self.generate_timetable()
        self.num_events = len(self.timetable)
        global MAX_EPISODE_STEPS
        MAX_EPISODE_STEPS = self.num_events
        
        self.current_index = 0
        # For each bus, track: location, next_available_time, used flag.
        self.bus_status = {bus_id: {"location": DEPOT, "next_available_time": OPERATION_START_MIN, "used": False} 
                           for bus_id in range(INITIAL_NUM_BUSES)}
        self.schedule = {bus_id: [] for bus_id in range(INITIAL_NUM_BUSES)}

    def generate_timetable(self):
        events = []
        for line_id, info in BUS_LINES.items():
            interval = info["interval"]
            t = OPERATION_START_MIN
            while t <= OPERATION_END_MIN:
                event = {
                    "time": t,
                    "line_id": line_id,
                    "terminal": info["terminal"],
                    "trip_time": info["trip_time"],
                    "rest_time": info["rest_time"]
                }
                events.append(event)
                t += interval
        events.sort(key=lambda e: (e["time"], e["line_id"]))
        return events

    def reset(self):
        self.current_index = 0
        self.bus_status = {bus_id: {"location": DEPOT, "next_available_time": OPERATION_START_MIN, "used": False} 
                           for bus_id in range(INITIAL_NUM_BUSES)}
        self.schedule = {bus_id: [] for bus_id in range(INITIAL_NUM_BUSES)}
        return self.get_state()

    def get_state(self):
        """
        Build state vector:
         - position 0: normalized current event time (current event time / OPERATION_END_MIN)
         - position 1: normalized bus line id (line_id / number of bus lines)
         - positions 2 to 2+INITIAL_NUM_BUSES-1: for each bus, a continuous availability value:
           (current event time - next_available_time) / T_RANGE.
        """
        if self.current_index >= self.num_events:
            return np.zeros(STATE_DIM, dtype=np.float32)
        event = self.timetable[self.current_index]
        norm_time = event["time"] / OPERATION_END_MIN
        norm_line = event["line_id"] / len(BUS_LINES)
        state = np.zeros(STATE_DIM, dtype=np.float32)
        state[0] = norm_time
        state[1] = norm_line
        for i in range(INITIAL_NUM_BUSES):
            # Availability value: positive means available; negative means not ready.
            avail_value = (event["time"] - self.bus_status[i]["next_available_time"]) / T_RANGE
            state[2 + i] = avail_value
        return state

    def compute_deadhead_cost(self, current_location, required_location):
        x1, y1 = COORDINATES[current_location]
        x2, y2 = COORDINATES[required_location]
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        normalized_cost = distance / MAX_DISTANCE
        return normalized_cost

    def step(self, action):
        if self.current_index >= self.num_events:
            return self.get_state(), 0, True, {}

        event = self.timetable[self.current_index]
        event_time = event["time"]
        required_terminal = event["terminal"]

        bus_info = self.bus_status[action]
        current_location = bus_info["location"]

        # --- Action Masking should already ensure that the selected bus is available,
        # but we also include a penalty for unavailability as safety.
        penalty_unavail = 0.0
        if bus_info["next_available_time"] > event_time:
            # Use configured penalty for buses not ready in time
            penalty_unavail = W_UNAVAILABILITY

        # Deadhead cost: if bus is not at required terminal.
        if current_location != required_terminal:
            deadhead_cost = self.compute_deadhead_cost(current_location, required_terminal)
        else:
            deadhead_cost = 0.0

        # Unused bus penalty (rn): if chosen bus is unused while another used bus is available at the depot.
        rn = 0.0
        if (not bus_info["used"]) and any(self.bus_status[b]["used"] and 
                                          self.bus_status[b]["location"] == DEPOT and 
                                          self.bus_status[b]["next_available_time"] <= event_time
                                          for b in self.bus_status):
            rn = 1.0

        # Chain bonus (W_CHAIN): if the chosen bus has been used before and its last event was on the same bus line.
        chain_bonus = 0.0
        if self.schedule[action]:
            last_event = self.schedule[action][-1]
            if last_event["line_id"] == event["line_id"]:
                chain_bonus = W_CHAIN

        # Rest reward (rk): if bus is used and available.
        rk = 1.0 if (bus_info["used"] and bus_info["next_available_time"] <= event_time) else 0.0
        ru = 0.0  # demand penalty (not implemented here)

        step_reward = - (W_UNUSED_PENALTY * rn + W_DEADHEAD * deadhead_cost + penalty_unavail) \
                      + W_REST_REWARD * rk + chain_bonus - W_DEMAND_PENALTY * ru

        # Record event and update bus status.
        self.schedule[action].append(event)
        self.bus_status[action]["used"] = True
        self.bus_status[action]["location"] = required_terminal
        self.bus_status[action]["next_available_time"] = event_time + event["trip_time"] + event["rest_time"]

        self.current_index += 1
        done = (self.current_index >= self.num_events)
        next_state = self.get_state()

        if done:
            num_used = sum(1 for b in self.bus_status if self.bus_status[b]["used"])
            final_penalty = -W_FINAL * num_used
            step_reward += final_penalty

        info = {"event": event, "deadhead_cost": deadhead_cost, "rn": rn, "penalty_unavail": penalty_unavail, "rk": rk, "chain_bonus": chain_bonus}
        return next_state, step_reward, done, info

    def print_and_save(self, text, file):
        """
        Helper function to print text to the console and write it to a file.
        """
        print(text, end="")  # Print to console
        file.write(text)     # Write to file

    def print_problem(self, file_path="data/results.txt"):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            self.print_and_save("=== Problem Definition ===\n", f)
            start_hr, start_min = divmod(OPERATION_START_MIN, 60)
            end_hr, end_min = divmod(OPERATION_END_MIN, 60)
            self.print_and_save(f"Operation Period: {start_hr:02d}:{start_min:02d} to {end_hr:02d}:{end_min:02d}\n\n", f)
            self.print_and_save("Bus Lines:\n", f)
            for line_id, info in BUS_LINES.items():
                self.print_and_save(f"  Bus Line {line_id}: {info['name']} - Loop at {info['terminal']}, "
                                    f"Interval: {info['interval']} min (Trip: {info['trip_time']} + Rest: {info['rest_time']})\n", f)
            self.print_and_save("\nTimetable (Departure Events):\n", f)
            for event in self.timetable:
                hr, minute = divmod(event["time"], 60)
                self.print_and_save(f"  {hr:02d}:{minute:02d} - Bus Line {event['line_id']} (Terminal: {event['terminal']})\n", f)
            self.print_and_save("==========================\n\n", f)

    def print_solution(self, file_path="data/results.txt"):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "a") as f:  # Append to the same file
            self.print_and_save("=== Final Bus Schedules (Solution) ===\n", f)
            for bus_id, events in self.schedule.items():
                if events:
                    self.print_and_save(f"Bus {bus_id} schedule:\n", f)
                    for event in events:
                        hr, minute = divmod(event["time"], 60)
                        self.print_and_save(f"  {hr:02d}:{minute:02d} - Bus Line {event['line_id']} (Terminal: {event['terminal']})\n", f)
                else:
                    self.print_and_save(f"Bus {bus_id} was not used.\n", f)
            self.print_and_save("========================================\n", f)
