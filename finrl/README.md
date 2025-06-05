
# 📰 FinRL Prediction Pipeline

The `/finrl` folder contains the python scripts and files necessary for the execution of the finrl prediction pipeline as part of the project.
The basis of this folder is the FinRL library provided by [AI4Finance](https://github.com/AI4Finance-Foundation/FinRL).
This pipeline aims to get near 'real-time' trading predictions for Agent 1 and Agent 2 on the latest data for **Nvidia stock**, whereby Agent 1 only has access to stock price data in its trading environment, Agent 2 has access to stock price data **and** sentiment risk scores of **Nvidia stock** related news obtained by the `/sentiment` part of this project.
The goal is to see Agent 2 perform better in its trades, given that it has additional up-to-date information on the conditions of the stock market.

---

## 🔍 Purpose

There are 5 principal python scripts in this directory. The purpose of each script is as follows:

### `processing.py`
 - Downloads hourly NVDA stock data from yahoo finance 600 days ago from the current date.
 - Preprocesses and combines the data with stock indicators using FinRL's FeatureEngineer
 - Splits the data into train/trade set (80%-20%) and saves them into `train_data.csv` and `trade_data.csv` for the next stage

### `training.py`
 - Trains the A2C and SAC model using `train_data.csv` obtained from `processing.py`
 - Called only if it is necessary to retrain the model
 - Otherwise, the pre-trained A2C and SAC model in `trained_models/` which was trained on the stock price data between 2023-10-02 to 2025-01-24 will be used.

### `utils.py`
 - Contains all the necessary functions needed in `inference.py`

### `inference.py`
 - Loads the `train_data.csv` and `trade_data.csv` obtained from `processing.py`, and loads the `aggregated_risk_scores.csv` from the `/sentiment` part
 - Runs the model prediction for Agent 1 (only stock data) and Agent 2 (stock data + sentiment data)
 - Calculates the MVO and loads the DJIA benchmark index hourly data as a base comparison.
 - Merges the results from Agent 1, Agent 2, MVO, and DJIA into results.csv to be plotted in the final dashboard.

### `main.py`
 - The main script that the workflow runs on
 - Calls the functions in order from `processing.py` > `training.py` > `inference.py`

---

## 📦 Features

### ➤ ⚖️ **Custom Trading Environment Class for Agent 2**  
 - `custom_env.py` shows a custom class built for Agent 2's Trading Environment to take the news sentiment risk scores into consideration during its trading steps.
 - This class inherits the StockTradingEnv class from **FinRL**, and only overwrittes the step() function to give 'reward' or 'penalty' to the agent depending on the risk scores. 
 - The amount of 'reward' or 'penalty' is indicated by a given weight value. 
 - For example, a lower risk score signals optimistic chance in its trade, thus the agent's state[0] is multiplied with a higher weight ('reward').
 - state[0] is a normalized or scaled value that is part of the observation used by the agent.
 - This is to mimic the agent's actions in a trading environment where the 'imaginary' state space includes the sentiment risk scores, as it is not possible to directly train the models on a state space with sentiment risk scores, due to the limitations on obtaining news articles through API calls for more than a week in the past. 

### ➤ ⚙️ **Automated Execution with bash script**  
 - Runs every Thursday at 12:00 PM.
 - Setup of automated run time in `crontab -e`.

Previously automated on **Github Actions**.

---

### 🚀 How to run the script manually
First, make sure you are in the right directory:

```bash
cd finrl
```

Make sure you are using Python version 3.11 and above:

```bash
python --version
```

To install the required libraries, run the following command:

```bash
pip install -r requirements.txt
```

To run the main script, run the following command:

```bash
python main.py
```

To run the finrl part in the automated bash script in the event of manual run or rerun from failed executions, run the following command:

```bash
bash start_script.sh finrl
```

## 📚 Further Reading
For more information on how FinRL works, check out their [Github Repo](https://github.com/AI4Finance-Foundation/FinRL) and [tutorials](https://github.com/AI4Finance-Foundation/FinRL-Tutorials).