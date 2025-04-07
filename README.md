# 🌀 Single Pendulum Digital Twin

A digital twin simulation of a single rod pendulum built with Python and [pygame](https://www.pygame.org/).  
This project models and simulates pendulum dynamics and includes real-world dataset collection, automated movement, and filtering experiments.

---

## 📦 Install

```bash
pip install -r requirements.txt
```

---

## 🚀 Run the Simulation

```bash
python simulate_pendulum.py
```

This launches the pendulum simulation UI built with `pygame`.  
It visualizes the dynamics of a single rod pendulum and is synced with real-world data.

---

## 🤖 Automated Movements

We implemented scripts to automatically move the cart and swing the pendulum upward.

![Pendulum Automated Movement](docs/automate_pendulum.gif)

## 📂 Dataset Collection

We collected real-world sensor data across different scenarios to calibrate and validate the simulation.  
Here are sample recordings of the data collection sessions:

### 🎥 Cart Movements Dataset

Logs accelerometer + encoder data as the cart moves in short bursts.

[🎥 Single Pendulum Movements Video 1](https://drive.google.com/file/d/1uwXgxUEdPyjAtTXvZwFLFt8k1ZoeykUe/view?usp=sharing)

[🎥 Single Pendulum Movements Video 2](https://drive.google.com/file/d/1cl6_7ZlVZOXNdCStdQezF6OfOyYDlMJa/view?usp=sharing)

---

### 🎥 Pendulum Extremes Dataset

Captures full-range theta swings — both minimum and maximum angles.

---

### 🎥 Single Pendulum Movements Dataset

Focuses on the pendulum's motion while the cart is stationary.

[🎥 Single Pendulum Movements Video](https://drive.google.com/file/d/18_DJQiT9Qgbnzhusx09J4R-xPQCMkzNB/view?usp=sharing)

## 🛠️ Built With

- Python 3.x
- Pygame
- NumPy
- Pandas
- SciPy
