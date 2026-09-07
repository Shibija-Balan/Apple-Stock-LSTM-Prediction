# Apple Stock LSTM Prediction

A time-series forecasting project using Long Short-Term Memory (LSTM) neural networks to model Apple stock prices from 2020 to 2025. The project intentionally preserves the original price-level LSTM, evaluates why it failed against a simple benchmark, and then redesigns the forecasting problem around next-day returns, multivariate market features and chronological validation.

## Project overview

The project began with a stacked LSTM that used 60-day rolling sequences of Apple mid-prices to predict the next trading day's mid-price. Although the predicted-price curve visually tracked the overall price trend, rigorous out-of-sample evaluation showed that the model underperformed a naive persistence forecast.

Rather than removing the failed specification, the project treats it as **Model 1**, diagnoses the weakness of price-level forecasting and then develops **Model 2**, a return-based multivariate LSTM designed to make the forecasting task more informative and the validation more rigorous.

---

# Model 1: Price-Level LSTM

## Architecture

- Target: next-day Apple mid-price
- Mid-price calculated as `(High + Low) / 2`
- 60-day rolling input window
- Two stacked LSTM layers with 70 units each
- 20% dropout after each LSTM layer
- Dense one-unit output layer
- Adam optimiser
- Mean-squared-error training loss
- 25 epochs
- Batch size of 32

## Leakage-free evaluation

`evaluate_model.py` reproduces the original architecture using a more rigorous out-of-sample process:

1. Loads a pinned public AAPL OHLC dataset covering 1 January 2020 to 17 January 2025
2. Calculates daily mid-price
3. Uses a chronological 80/20 train-test split
4. Fits the MinMax scaler only on the training period to prevent preprocessing leakage
5. Trains the same two-layer LSTM architecture
6. Evaluates predictions in original US-dollar price units
7. Reports RMSE, MAE and directional accuracy
8. Benchmarks against a naive persistence forecast where tomorrow's predicted price equals today's observed price

## Model 1 results

Evaluation window: 1,264 daily observations, with 1,011 training observations and 253 test observations.

| Metric | Model 1 LSTM | Naive persistence |
|---|---:|---:|
| RMSE | **$6.23** | **$2.80** |
| MAE | **$5.17** | **$2.00** |
| Directional accuracy | **49.8%** | n/a |

Additional results:

- Mean test-period mid-price: **$208.16**
- Normalised RMSE: **3.0%** of the mean test-period mid-price
- RMSE skill vs persistence: **-122.9%**
- MAE skill vs persistence: **-158.6%**

### Diagnosis

The original model visually follows Apple's price trend, but this is not sufficient evidence of predictive value. Stock-price levels are highly persistent, so a model can appear to track the series while still performing worse than the simple assumption that tomorrow's price will be approximately today's price.

Model 1 therefore provided two important findings:

- the original LSTM did not demonstrate useful next-day directional forecasting ability, with directional accuracy close to chance at 49.8%;
- a complex neural-network model should not be judged from a predicted-vs-actual graph alone and must be benchmarked against a simple baseline.

---

# Model 2: Return-Based Multivariate LSTM

## Redesign rationale

Instead of predicting the absolute next-day price level, Model 2 predicts the next-day simple return:

`r(t+1) = (P(t+1) - P(t)) / P(t)`

This reframes the problem from learning the highly persistent price level to predicting the next day's movement relative to today's price.

The predicted return is then converted back into a price forecast:

`Predicted Price(t+1) = P(t) * (1 + Predicted Return(t+1))`

This allows Model 2's price RMSE to remain directly interpretable in dollars and comparable with the price-level model.

## Feature engineering

Model 2 uses ten features derived only from information available at time `t`:

- 1-day return
- 5-day return
- 10-day momentum
- 5-day rolling volatility
- 20-day rolling volatility
- daily high-low range as a percentage of mid-price
- open-close percentage movement
- daily volume change
- 5-day moving-average gap
- 20-day moving-average gap

## Validation design

The redesigned experiment uses a chronological:

- **70% training set**
- **15% validation set**
- **15% test set**

Feature and target scalers are fitted on the training period only. The validation set is used for model selection and early stopping, while the test period remains untouched until final evaluation.

## Model 2 architecture

- Target: next-day return
- 20-day multivariate sequence
- 10 engineered input features
- LSTM layer with 32 units
- 20% dropout
- Dense layer with 16 ReLU units
- Dense one-unit output layer
- Adam optimiser with learning rate `5e-4`
- Mean-squared-error loss
- Maximum 60 epochs
- Early stopping with best weights restored
- Batch size of 32
- No shuffling of chronological observations

The smaller architecture was chosen to reduce unnecessary model complexity relative to the limited number of daily observations.

## Model 2 results

| Metric | Model 1 | Model 2 |
|---|---:|---:|
| Price RMSE | **$6.23** | **$2.82** |
| Price MAE | **$5.17** | **$1.89** |
| Directional accuracy | **49.8%** | **66.3%** |

Model 2's persistence benchmark on the same final test period achieved approximately **$3.00 RMSE**.

Therefore:

- Model 2 reduced reported price RMSE by approximately **54.8%** relative to Model 1
- directional accuracy increased from **49.8% to 66.3%**
- Model 2 achieved approximately **6.1% lower RMSE than persistence on the same Model 2 test period**
- early stopping selected approximately epoch **17** as the best validation point

### Important comparison note

The direct Model 1-to-Model 2 RMSE reduction should be interpreted as a project-level before/after comparison rather than a perfectly controlled model comparison because the two reported evaluations use different holdout designs. The more important benchmark for Model 2 is the persistence forecast calculated on the **same Model 2 test period**, where Model 2 achieved positive RMSE skill.

## Interpretation

The redesign produced a substantially stronger forecasting experiment. The improvement came from changing the target, adding features related to momentum and volatility, reducing model complexity and separating validation from final testing rather than simply increasing the number of LSTM units.

The project therefore demonstrates an iterative modelling workflow:

**prototype -> benchmark -> diagnose failure -> redesign target and features -> validate -> test -> compare against baseline**

The result should not be interpreted as proof of a profitable trading strategy. Directional accuracy and price-error metrics do not incorporate transaction costs, turnover, position sizing or risk. Strategy-level statistics such as Sharpe ratio would require a separately defined trading rule and backtest.

---

## Reproducible evaluation scripts

### Model 1

```bash
python evaluate_model.py
```

Produces `metrics.json` with:

- LSTM RMSE and MAE in USD
- persistence RMSE and MAE
- RMSE and MAE skill versus persistence
- directional accuracy
- normalised RMSE
- training loss
- observation counts and modelling parameters

### Model 2

```bash
python evaluate_improved_model.py
```

Produces `improved_metrics.json` with:

- Model 2 price RMSE and MAE
- return RMSE
- persistence benchmark on the same test period
- directional accuracy
- RMSE reduction relative to Model 1
- skill versus persistence
- chronological split information
- best epoch selected by validation loss

## Notebook

`Apple_Stock_LSTM.ipynb` contains both stages of the project:

1. the original exploratory price-level LSTM and its visualisations;
2. a new **Model 2: Improving the Model** section documenting the failure diagnosis, feature engineering, return-based target, revised validation design, redesigned LSTM and final comparison.

## Visualisations

The notebook includes:

- Apple mid-price history
- Model 1 predicted vs actual prices
- Model 1 training loss
- Model 2 predicted vs actual prices
- Model 2 training and validation loss
- Model 1 / Model 2 / persistence comparison outputs

## Repository structure

- `Apple_Stock_LSTM.ipynb` — full project notebook containing Model 1 and Model 2
- `evaluate_model.py` — leakage-free Model 1 evaluation
- `evaluate_improved_model.py` — return-based Model 2 evaluation
- `metrics.json` — Model 1 evaluation output when generated
- `improved_metrics.json` — Model 2 evaluation output when generated
- `.github/workflows/evaluate.yml` — automated Model 1 evaluation
- `.github/workflows/evaluate_improved.yml` — automated Model 2 evaluation
- `images/` — project visualisations
- `requirements.txt` — Python dependencies

## Tools

Python, TensorFlow/Keras, pandas, NumPy, matplotlib, scikit-learn, Git and GitHub Actions

## Key learning

- Preparing financial time-series data for supervised sequence models
- Building and training LSTM networks
- Preventing preprocessing leakage
- Using chronological train/validation/test splits
- Engineering return, momentum and volatility features
- Using early stopping to control overfitting
- Evaluating forecasts in economically interpretable price units
- Benchmarking complex models against persistence
- Distinguishing visual fit from genuine out-of-sample predictive improvement
- Iterating on a failed model rather than hiding poor benchmark performance

## CV-ready project description

Built and iteratively improved an LSTM-based Apple stock forecasting model, diagnosing a $6.23-RMSE price-level specification that underperformed persistence and redesigning it around next-day returns, multivariate momentum/volatility features and chronological validation; reduced reported price RMSE to $2.82 and increased directional accuracy from 49.8% to 66.3%, with the redesigned model outperforming persistence by 6.1% RMSE on its final holdout period.
