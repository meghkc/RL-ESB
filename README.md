# Electric Bus Scheduling with Deep Reinforcement Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Deep Reinforcement Learning solution for optimizing electric bus fleet scheduling with energy management, time-of-use pricing, and curriculum learning.

## 🎯 Features

- **PPO-based DRL Agent**: Proximal Policy Optimization for stable learning
- **Curriculum Learning**: Progressive training across fleet sizes (25→40→50+ buses)
- **Energy Management**: Battery state-of-charge optimization and charging strategies
- **Time-of-Use Pricing**: Dynamic electricity cost optimization
- **Real-world Constraints**: Depot scheduling, trip timing, and energy consumption

## 🚀 Quick Start

### Prerequisites

```bash
pip install torch numpy pandas matplotlib
```

### Installation

```bash
git clone https://github.com/yourusername/electric-bus-scheduling-drl.git
cd electric-bus-scheduling-drl
pip install -r requirements.txt
```

### Usage

#### 1. Training with Curriculum Learning

```bash
python main.py --mode curriculum
```

#### 2. Training Standard PPO

```bash
python main.py --mode train --episodes 50000
```

#### 3. Evaluation

```bash
python main.py --mode evaluate --model_path models/model_final.pth
```

#### 4. Configuration Validation

```bash
python main.py --mode validate
```

## 📊 Project Structure

```
electric-bus-scheduling-drl/
├── main.py               # Main entry point
├── config.py             # Configuration and hyperparameters
├── environment.py        # RL environment implementation
├── ppo_agent.py          # PPO agent implementation
├── curriculum_learning.py # Curriculum learning implementation
├── utils.py              # Utility functions
├── data/                 # Data files
│   └── L4node5_10_15.csv # Sample timetable (place your CSV here)
├── models/               # Trained models
├── tests/                # Unit tests
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🛠️ Configuration

Edit `config.py` to customize:

- **Fleet Parameters**: Number of buses, battery capacity
- **Energy Settings**: Consumption rates, charging speeds
- **Pricing**: Time-of-use electricity rates
- **RL Hyperparameters**: Learning rate, episodes, network architecture

## 📈 Results

The system learns to:
- Minimize total fleet size while meeting all trip demands
- Optimize charging schedules to reduce electricity costs
- Balance energy consumption with operational constraints
- Handle scheduling conflicts and peak demand periods

## 🔬 Technical Details

### Environment State Space
- Current time and operational status
- Bus positions and energy levels
- Trip queue and scheduling information

### Action Space
- Bus assignments to trips
- Charging decisions at depot

### Reward Function
- Fleet size minimization
- Energy cost optimization
- Constraint violation penalties
- Time-of-use pricing incentives

## 📚 Documentation

For detailed documentation, see the code comments and docstrings in each module.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.

## 📞 Contact

Project Link: [https://github.com/yourusername/electric-bus-scheduling-drl](https://github.com/yourusername/electric-bus-scheduling-drl)

## 🙏 Acknowledgments

- Built with PyTorch and NumPy
- Inspired by advances in deep reinforcement learning for transportation
- Thanks to the open-source RL community
