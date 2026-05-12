# RiG-Sense: Risk- and Graph-Aware Sensor Placement for Contamination Detection in Water Distribution Networks

RiG-Sense is a Deep Reinforcement Learning-based sensor placement framework for contamination detection in Water Distribution Networks (WDNs). This project combines Deep Q-Network (DQN), grid search-based hyperparameter tuning, and risk-aware and graph-aware features to improve sensor placement quality under a proportional sensor budget.

The framework is designed to select sensor locations from candidate network links and evaluate their effectiveness using contamination detection coverage and detection time.

---

## 1. Clone Repository

Clone this repository from GitHub:

```bash
git clone https://github.com/Berteknologi-id/RiG-Sense.git
```

Move into the project directory:

```bash
cd rig-sense
```


---

## 2. Create Virtual Environment

It is recommended to use a virtual environment so that the required libraries for this project do not conflict with other Python projects.

### Windows

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment:

```bash
venv\Scripts\activate
```

### macOS / Linux

Create a virtual environment:

```bash
python3 -m venv venv
```

Activate the virtual environment:

```bash
source venv/bin/activate
```

After activation, the terminal should show the virtual environment name:

```bash
(venv)
```

---

## 3. Upgrade pip

Before installing the required libraries, upgrade `pip`:

```bash
python -m pip install --upgrade pip
```

---

## 4. Install Required Libraries

Install the required Python libraries using the following command:

```bash
pip install numpy pandas matplotlib tensorflow gym openpyxl
```

The main libraries used in this project are:

```text
numpy
pandas
matplotlib
tensorflow
gym
openpyxl
```

The following libraries are built-in Python modules and do not need to be installed separately:

```text
os
random
ast
time
datetime
collections
itertools
typing
dataclasses
```

---

## 5. Install Dependencies Using requirements.txt

You can also install all dependencies from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

Example content of `requirements.txt`:

```txt
numpy
pandas
matplotlib
tensorflow
gym
openpyxl
```

For a more stable and reproducible environment, you can use versioned dependencies:

```txt
numpy==1.26.4
pandas==2.2.3
matplotlib==3.9.2
tensorflow==2.15.1
gym==0.26.2
openpyxl==3.1.5
```

---

## 6. Recommended Python Version

The recommended Python version for this project is:

```text
Python 3.10
```

Python 3.10 is recommended because it provides good compatibility with TensorFlow, Gym, NumPy, Pandas, and Matplotlib.

Check your Python version using:

```bash
python --version
```

or:

```bash
python3 --version
```

---


## 7. Citation

If you use this repository in your research, please cite the related paper:

```text
RiG-Sense: Risk- and Graph-Aware Sensor Placement Strategy for Contamination Detection in Water Distribution Networks.
```

---

## 8. License

This project is intended for academic and research purposes.
