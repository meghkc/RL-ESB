# Electric Bus Scheduling Optimization with Deep Reinforcement Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive Deep Reinforcement Learning solution for optimizing electric bus fleet scheduling with multiple state-of-the-art on-policy & off-policy algorithms. This project implements and compares four different RL approaches: **PPO**, **DDQN**, **REINFORCE**, and **A2C** for solving the electric bus scheduling problem.

## Project Lineage

This repository is inherited from the original project:
- Source: [https://github.com/meghkc/RL-ESB](https://github.com/meghkc/RL-ESB)

The original codebase provides the baseline RL environment, agents, and training scripts for a synthetic electric bus scheduling setup.

## What Changed In This Repo

This fork extends the original project with dataset-driven scheduling support and related training updates:

- Added `dataArticleJuliette/` support (Montreal transit benchmark instance format from here https://www.gerad.ca/%7eguyd/mdevsp.html).
- Added `dataset_loader.py` to parse `depots.txt`, `recharge.txt`, `voyages.txt`, and `hlp.txt`.
- Updated `environment.py` to support both:
  - `synthetic` mode (original config-based setup), and
  - `juliette` mode (instance files from `dataArticleJuliette`).
- Updated `run.py` with CLI options for dataset selection:
  - `--data-source`, `--dataset-root`, `--dataset-subset`, `--dataset-split`, `--dataset-instance`.
- Updated PPO flow (`run.py`, `ppo_agent.py`) to use environment-provided valid actions and dynamic state/action dimensions.
- Updated setup notes so only core PyTorch is required by default (no mandatory `torchvision`/`torchaudio`).

Added dataset CLI support so PPO can run on a specific Juliette instance (e.g., python run.py --data-source juliette --dataset-root dataArticleJuliette --dataset-subset A --dataset-split Training --dataset-instance Network9a_22_0).

### Simulator Environment with Juliette Dataset

#### Overview

The core class is `BusSchedulingEnv` in `environment.py`. It models a **sequential assignment problem**: at each step, the agent assigns one of the available buses to the next trip in the timetable (sorted by departure time).

> **Note:** This is a planning/scheduling task, not real-time control. The full timetable is known upfront and the environment is fully deterministic — the agent learns a constructive heuristic that generalizes across problem instances.

#### Observations

Observation vector dimension: `2 + num_buses`

| Index | Meaning | Normalization |
|-------|---------|---------------|
| `[0]` | Current trip departure time | `(event_time - op_start) / t_range` → [0, 1] |
| `[1]` | Bus line ID of the current trip | `(line_idx - 1) / (num_lines - 1)` → [0, 1] |
| `[2..2+N]` | Per-bus readiness for each of the N buses | `(event_time - (bus.next_available_time + deadhead_to_trip)) / t_range` |

The per-bus readiness is the key signal: **positive** means the bus is available at the trip's origin on time; **negative** means it would arrive late (infeasible without penalty).

#### Juliette Dataset Files

Four files are parsed by `dataset_loader.py`:

| File | Content | Used For |
|------|---------|---------|
| `depots.txt` | `depot_node; num_buses` | Initial bus locations |
| `voyages.txt` | `trip_id; from_node; dep_time; to_node; arr_time; line_id` | Timetable of trips to schedule |
| `hlp.txt` | `from_node; to_node; travel_time` (precomputed shortest paths) | Deadhead travel times between nodes |
| `recharge.txt` | Charging station locations | Loaded but not yet factored into reward |

All times are in **minutes from start of day**.

#### Travel Times

The simulator is **fully deterministic** — no stochasticity of any kind:

- **Scheduled trip durations** come directly from `voyages.txt` (arrival − departure). Always fixed.
- **Deadhead times** are precomputed shortest-path lookups from `hlp.txt`. The same origin → destination pair always yields the same travel time.
- No Gaussian noise, delay distributions, or traffic perturbations are applied.

## Features

- **Multi-Algorithm Implementation**: Compare 4 different RL approaches
  - **PPO (Proximal Policy Optimization)**: Stable policy gradient method
  - **DDQN (Double Deep Q-Network)**: Advanced value-based method
  - **REINFORCE**: Classic policy gradient algorithm
  - **A2C (Advantage Actor-Critic)**: Actor-critic with advantage estimation
- **GPU/CPU Support**: Automatic hardware detection and optimization
- **Comprehensive Comparison**: Side-by-side algorithm performance analysis
- **Real-world Constraints**: Multi-line bus scheduling with time constraints
- **Action Masking**: Intelligent constraint handling for valid bus assignments
- **Reward Engineering**: Multi-objective optimization (fleet size, deadhead costs, chain bonuses)

## Quick Start

### Prerequisites

```bash
pip install torch numpy matplotlib pandas tqdm
# Optional only if your project code needs them:
# pip install torchvision torchaudio
```

### Installation

```bash
git clone https://github.com/meghkc/RL-ESB.git
cd RL-ESB
pip install -r requirements.txt
```

### Usage

#### 1. Train Individual Algorithms

```bash
# Train PPO
python run.py

# Train DDQN  
python run_ddqn.py

# Train REINFORCE
python run_reinforce.py

# Train A2C
python run_a2c.py
```

#### 2. Compare All Algorithms

```bash
# Run comparison analysis and generate plots
python compare_algorithms.py
```

#### 3. Algorithm-Specific Training Scripts

```bash
# PPO with custom parameters
python run.py --episodes 1000

# DDQN with exploration tuning
python run_ddqn.py

# REINFORCE policy gradient
python run_reinforce.py  

# A2C actor-critic
python run_a2c.py
```

## Project Structure

```
RL-ESB/
|-- config.py                    # System configuration and hyperparameters
|-- environment.py               # Bus scheduling environment implementation
|
|-- # Algorithm Implementations
|-- ppo_agent.py                 # Proximal Policy Optimization agent
|-- ddqn_agent.py                # Double Deep Q-Network agent
|-- reinforce_agent.py           # REINFORCE policy gradient agent
|-- a2c_agent.py                 # Advantage Actor-Critic agent
|
|-- # Training Scripts
|-- run.py                       # PPO training and evaluation
|-- run_ddqn.py                  # DDQN training and evaluation
|-- run_reinforce.py             # REINFORCE training and evaluation
|-- run_a2c.py                   # A2C training and evaluation
|-- advanced_comparison.py       # Multi-algorithm comparison and analysis
|
|-- data/                        # Results and model storage
|   |-- *_training.txt           # Training progress logs
|   |-- *_results.txt            # Evaluation results
|   |-- *_best_model.pth         # Best trained models
|   `-- figures/                 # Training plots and comparisons
|       |-- *_training_rewards.png
|       `-- algorithm_comparison.png
|
|-- requirements.txt             # Python dependencies
`-- README.md                    # This file
```

## Configuration

Edit `config.py` to customize:

- **Bus Lines**: Multiple bus routes with different terminals and schedules
- **Operation Parameters**: Operating hours (6 AM - 9 PM), trip times, rest periods
- **Fleet Settings**: Number of available buses, depot configuration
- **Reward Weights**: Deadhead costs, unused bus penalties, chain bonuses
- **Algorithm Hyperparameters**: Learning rates, network architectures, training episodes

## Algorithm Comparison

| Algorithm | Type | Key Features | Best For |
|-----------|------|--------------|----------|
| **PPO** | Policy Gradient | Stable, clipped updates | General purpose, stable training |
| **DDQN** | Value-based | Experience replay, target network | Sample efficiency, discrete actions |
| **REINFORCE** | Policy Gradient | Simple, direct optimization | Understanding baselines |
| **A2C** | Actor-Critic | Advantage estimation, shared features | Reduced variance, faster convergence |

## Results

The system learns to:
- **Minimize Fleet Size**: Use fewer buses while meeting all trip demands
- **Optimize Scheduling**: Reduce deadhead movements between terminals  
- **Create Efficient Chains**: Assign consecutive trips to same buses on same lines
- **Handle Constraints**: Respect bus availability and time windows
- **Balance Trade-offs**: Fleet size vs. operational efficiency

### Performance Metrics
- **Average Reward**: Overall system performance
- **Buses Used**: Fleet efficiency (typically 3-5 buses from 10 available)
- **Training Stability**: Convergence speed and consistency
- **Solution Quality**: Meeting all trip demands with minimal resources

## Technical Details

### Problem Definition
- **3 Bus Lines**: Different terminals (Terminal1, Terminal2, Terminal3) with varying schedules
- **Operation Period**: 15-hour daily operation (6:00 AM - 9:00 PM)
- **Fleet Management**: 10 available buses, minimize actual usage
- **Scheduling Constraints**: Bus availability, terminal locations, trip timing

### Environment State Space
- **Time Information**: Normalized current event time and bus line ID
- **Bus Status**: Availability times for each bus (positive = available, negative = busy)
- **Location Tracking**: Current terminal positions for deadhead cost calculation

### Action Space
- **Bus Assignment**: Select which bus (0-9) to assign to each trip
- **Action Masking**: Only available buses can be selected
- **Constraint Handling**: Automatic filtering of invalid assignments

### Reward Function Components
- **Deadhead Cost**: Penalty for moving empty buses between terminals
- **Unused Bus Penalty**: Discourage using new buses when used buses are available  
- **Rest Reward**: Bonus for reusing available buses
- **Chain Bonus**: Extra reward for consecutive trips on same line
- **Final Penalty**: Total fleet size minimization at episode end

### Algorithm Implementations

#### PPO (Proximal Policy Optimization)
- **Architecture**: Shared state network + separate actor/critic heads
- **Features**: Action masking, GAE advantages, clipped policy updates
- **Stability**: Trust region optimization for consistent learning

#### DDQN (Double Deep Q-Network)  
- **Architecture**: Separate main and target Q-networks
- **Features**: Experience replay, epsilon-greedy exploration, target network updates
- **Efficiency**: Sample reuse and stable value estimation

#### REINFORCE
- **Architecture**: Simple policy network with softmax output
- **Features**: Monte Carlo returns, advantage normalization
- **Simplicity**: Direct policy gradient optimization

#### A2C (Advantage Actor-Critic)
- **Architecture**: Shared base + actor/critic heads
- **Features**: TD advantage estimation, simultaneous policy/value updates
- **Balance**: Reduced variance while maintaining policy gradient benefits

## Hardware Support

All algorithms automatically detect and utilize available hardware:

- **GPU Acceleration**: CUDA support for faster training
- **CPU Fallback**: Automatic CPU usage when GPU unavailable  
- **Device Logging**: Clear indication of hardware being used
- **Memory Optimization**: Efficient tensor operations and batch processing

## Visualization and Analysis

The project generates comprehensive visualizations:

- **Individual Training Curves**: Progress for each algorithm
- **Comparative Analysis**: Side-by-side performance comparison
- **Performance Metrics**: Detailed evaluation statistics
- **Solution Quality**: Bus usage efficiency and constraint satisfaction

## Documentation

For detailed documentation, see:
- Code comments and docstrings in each module
- Algorithm-specific implementation details
- Environment dynamics and reward engineering
- Comparison methodology and metrics

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License.

## Contact

Project Link: [https://github.com/meghkc/RL-ESB](https://github.com/meghkc/RL-ESB)

## Acknowledgments

- Built with PyTorch for deep learning and GPU acceleration
- Inspired by advances in deep reinforcement learning for transportation optimization
- Algorithm implementations based on seminal papers in RL
- Thanks to the open-source RL community for foundational work
- Multi-algorithm comparison methodology for comprehensive evaluation

