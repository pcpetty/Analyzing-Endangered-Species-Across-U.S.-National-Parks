# Analyzing Endangered Species Across U.S. National Parks

## Introduction

### Objective
This project focuses on analyzing the distribution of endangered species across various U.S. national parks. By examining patterns in species conservation statuses, we aim to identify parks and species categories that are at the highest risk. Our analysis will use data on species observations and conservation statuses, applying statistical and machine learning techniques to draw insights and predictions about species endangerment.

### Dataset Overview
We use two primary datasets in this project:

- **Species Information Dataset:** Contains data on species names, categories (e.g., Mammals, Birds), and their conservation status (e.g., Endangered, Threatened).
- **Observations Dataset:** Contains data on species observations in different national parks, detailing the park name and observation count for each species.

### Key Questions
1. Which national parks have the highest concentration of endangered species?
2. Are there specific species categories (e.g., Mammals, Birds) that are more likely to be endangered?
3. Can we predict the conservation status of a species using features like species type and observation count?

---

## Project Objectives

- **Leverage Jupyter Notebook** to effectively communicate findings.
- **Analyze and visualize data** related to species conservation statuses in national parks.
- **Apply machine learning models** to predict species endangerment and assess their performance.
- **Develop actionable insights** that could support biodiversity conservation efforts.

---

## Understanding the Data

We begin by examining the dataset provided by the National Parks Service. The data includes information on various species, their conservation statuses, and the number of observations recorded across different national parks. Our initial focus is on cleaning and preparing the data to ensure accuracy in the subsequent analysis.

---

## Data Cleaning and Wrangling

Data cleaning is crucial for ensuring the reliability of our analysis. We addressed missing values in the `conservation_status` column by filling them with "No Intervention," reflecting species that are not currently under conservation watch. Additionally, we standardized text data in the `common_names` column to avoid discrepancies and ensure consistency throughout the dataset.

---

## Exploratory Data Analysis (EDA)

### 1. Distribution of Conservation Statuses Across Parks
- **Heatmap Visualization:** A heatmap visualizes the proportion of different conservation statuses across national parks. The visualization indicates that certain parks have a higher concentration of endangered species, suggesting the need for focused conservation efforts.

### 2. Correlations Between Species Categories and Conservation Status
- **Contingency Table and Chi-Square Test:** A contingency table shows the counts of different species categories across each conservation status. The Chi-square test results reveal a strong association between species categories and their conservation statuses, suggesting that the type of species significantly influences its likelihood of being endangered.

---

## Machine Learning Analysis

### 1. Logistic Regression
Using logistic regression, we predict the likelihood of a species being endangered based on features like species category and observation count. This model helps us understand the key factors contributing to species endangerment.

### 2. Decision Tree and Random Forest Classifiers
These models are implemented to predict the conservation status of species. They provide insights into the most influential features, such as species type and park location, in determining conservation status. The model's performance is evaluated using a confusion matrix.

---

## Feature Engineering

To enhance our models, we explored interaction features by combining existing variables to capture more complex relationships. Although these features did not significantly improve model performance, the exercise demonstrated the potential for deeper insights through advanced feature engineering techniques.

---

## Conclusion

This project provided valuable insights into the biodiversity and conservation efforts within U.S. national parks. Our analysis identified key parks and species categories that require attention, while our models successfully predicted species endangerment with high accuracy. Future work could explore additional data sources or more advanced modeling techniques to further refine these predictions.

---

## Next Steps

Looking ahead, there are several opportunities to extend this analysis:

- **Incorporating Time-Series Data:** Analyzing trends in species conservation statuses over time to identify emerging threats.
- **Expanding the Feature Set:** Including environmental factors such as climate and habitat data to improve model accuracy.
- **Exploring Other Algorithms:** Applying ensemble methods or deep learning models for more sophisticated predictions.

