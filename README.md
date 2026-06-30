# AI Automation Impact Prediction

This project aims to predict the 'AI Automation Impact' score using machine learning models, based on a dataset covering tech layoffs and hiring trends.

## Project Overview

The goal of this project is to build and evaluate predictive models that can estimate the potential impact of AI automation within various companies and industries. We explore different machine learning algorithms and compare their performance.

## Dataset

The analysis is performed on the `tech_layoffs_hiring_trends_elite_v2.csv` dataset, which includes information such as company details, layoff counts, AI impact scores, hiring trends, and market conditions.

## Key Features

*   **Data Loading & Exploration:** Initial examination of the dataset structure and summary statistics.
*   **Data Preprocessing:** Handling categorical features (One-Hot Encoding), numerical feature scaling (StandardScaler), and imputation of missing values.
*   **Model Training & Comparison:** Implementation of two regression algorithms:
    *   `RandomForestRegressor`
    *   `LinearRegression`
*   **Model Evaluation:** Assessment of model performance using metrics relevant to regression tasks (MAE, MSE, RMSE, R2 Score).
*   **Visualizations:** Creation of plots to visualize actual vs. predicted values, residual plots, and a comparison of model R2 scores.

## Results

Both `RandomForestRegressor` and `LinearRegression` showed meaningful predictive capabilities for 'AI Automation Impact'. Linear Regression slightly outperformed RandomForestRegressor in terms of R2 score, suggesting a strong underlying linear relationship in the data for this prediction task.

## How to Run

1.  Ensure you have Python installed with the necessary libraries (`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`).
2.  Place the `tech_layoffs_hiring_trends_elite_v2.csv` dataset in the same directory as the notebook.
3.  Execute the cells in the provided Jupyter/Colab notebook sequentially.

## Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
