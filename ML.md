# Preprocessing Techniques in Machine Learning Pipeline: Justification from EDA Findings

## Data Cleaning Rationale

### Removing SpecialReduction Feature
**EDA Evidence**: The feature correlation analysis during EDA revealed that 'SpecialReduction' had 0 correlation with the settlement value. Hence, it was sensible to remove it to reduce dimensionality without losing predictive power.

### Removing Zero/Null Settlement Values
**EDA Evidence**: The EDA showed that the settlement values ranged from £240 to £7,862.90 with the mean around £1,218.01. Removing zero/null values is critical because:
1. Zero settlement values would skew the model's predictions
2. The target variable should have valid values for supervised learning
3. The business context requires positive settlement values

### Removing Rows with Many Missing Features
**EDA Evidence**: The EDA demonstrated that 3,545 rows (71%) had complete data, while the remaining had varying degrees of missing values. The missing data analysis showed a pattern where rows with 5+ missing features represented less than 4% of the data. Removing these heavily incomplete records is justified as they could introduce noise while retaining 96.4% of the dataset preserves most of the information.

## Feature Engineering Justification

### Creating Days_Between_Accident_And_Claim
**EDA Evidence**: The temporal analysis in the EDA revealed:
- Clear seasonal patterns in accident frequency
- A consistent lag between accident date and claim filing date
- Potential correlation between timing and settlement values

Converting these dates into a numerical feature captures this important lag relationship that could influence settlement amounts.

### Converting Injury_Prognosis to Numeric
**EDA Evidence**: The EDA showed a strong relationship between injury prognosis duration and settlement values:
- Longer recovery periods (especially 12+ months) strongly correlated with higher settlements
- The relationship appeared somewhat linear between prognosis duration and settlement amount

Converting this from categorical format (e.g., "A. 1 month") to numeric values enables the model to capture this linear relationship.

## Train-Test Split Before Preprocessing

**Justification**: The pipeline performs the train-test split before preprocessing to prevent data leakage. This is critical because:

1. **Preventing Statistical Leakage**: If preprocessing (especially imputation and encoding) happened on the entire dataset first, information from the test set would influence the training process. For example:
   - Mean values for imputation would include test data information
   - Frequency counts for categorical encoding would include test data distributions

2. **Real-World Application**: In deployment, new data will arrive without known preprocessing parameters, so the model must learn to apply transformations based only on training data.

3. **Proper Validation**: This approach ensures that model evaluation on the test set truly represents its performance on unseen data.

## Imputation Strategies

### Mean Imputation for Numerical Features
**EDA Evidence**: The numerical variable distributions in the EDA showed:
- Most numerical features had moderate skewness
- The distributions were not extremely long-tailed
- Mean values likely represented the central tendency adequately

This makes mean imputation an appropriate choice for numerical features, balancing simplicity with reasonable accuracy.

### Most Frequent (Mode) Imputation for Categorical Features
**EDA Evidence**: The categorical feature analysis showed distinct distributions with clear dominant categories in many cases. For example:
- Dominant injury types had clear frequency patterns
- Accident types showed characteristic distributions

Using the most frequent value preserves the dominant patterns identified in the EDA while filling missing values.

## Encoding Techniques

### Label Encoder for Binary Features
**EDA Evidence**: The categorical analysis showed several binary features like:
- Minor_Psychological_Injury (Yes/No)
- Whiplash (Yes/No)
- Police Report Filed and Witness Present

These variables have only two values, making Label Encoding ideal as it creates a single feature (0/1) for each variable without unnecessary dimensionality expansion.

### One-Hot Encoder for Nominal Features
**EDA Evidence**: The EDA revealed nominal categorical variables with no inherent order, including:
- Gender
- Vehicle Type
- Weather Conditions

These have multiple non-ordinal categories that could each have different effects on the settlement value, requiring one-hot encoding to properly capture their influence.

### Target Encoder for High-Cardinality Features
**EDA Evidence**: The EDA highlighted features with many unique values:
- AccidentType had multiple categories with varying settlement impacts
- Injury descriptions contained detailed, variable text
- Dominant injury showed numerous possible values

Target encoding for these high-cardinality features prevents dimensionality explosion while maintaining predictive power by replacing categories with their relationship to the target variable.

## Correlation Analysis Methods

### Traditional Correlation Matrix
The heatmap shows linear relationships between features and the settlement value, which the EDA identified as being strong for features like:
- GeneralRest and SpecialTherapy (strong positive correlations with settlement)
- Injury prognosis duration

### Random Forest Feature Importance
This captures non-linear relationships that the EDA suggested existed, particularly:
- Interaction effects between injury types
- The non-linear relationship between certain accident types and settlement values

## Fit_Transform vs. Transform

The code uses `fit_transform()` on training data but only `transform()` on test data because:

1. The `fit` part learns parameters from the data (e.g., mean values for imputation, encodings for categories)
2. The `transform` part applies those learned parameters

Using only `transform()` on test data ensures the model applies the parameters learned from training data, preventing test data information from influencing the preprocessing parameters and ensuring the validation is realistic. Another measure to prevent data leakage to the models.

This approach maintains the integrity of the test set as truly unseen data, which is crucial for honest model evaluation.

# Machine Learning Models: Rationale and Performance Analysis for Insurance Claim Prediction

## Model Approaches and Reasoning

### Basic Machine Learning Models

1. **Decision Tree Regressor**
   - **Rationale**: Serves as an interpretable baseline model with a transparent decision-making process.
   - **Design**: Uses GridSearchCV to optimize hyperparameters like max_depth, min_samples_split, min_samples_leaf, and ccp_alpha.
   - **Purpose**: Provides clear splitting rules that help understand key feature thresholds and relationships in the insurance data.

2. **Random Forest Regressor**
   - **Rationale**: Improves on decision trees by reducing overfitting through ensemble learning.
   - **Design**: Uses an ensemble of trees with different bootstrap samples and random feature selection.
   - **Purpose**: Handles the complex, non-linear relationships between claim features and settlement values while offering feature importance measures.

3. **Gradient Boosting Regressor**
   - **Rationale**: Attempts to iteratively correct prediction errors from previous trees.
   - **Design**: Sequential building of trees where each new tree focuses on the residuals of previous trees.
   - **Purpose**: Excels at capturing subtle patterns in settlement values, particularly useful for the highly variable claim amounts.

4. **Extreme Gradient Boosting (XGBoost) Regressor**
   - **Rationale**: More regularized implementation of gradient boosting to further prevent overfitting.
   - **Design**: Uses more sophisticated regularization techniques (L1, L2) and optimized computation.
   - **Purpose**: Better handles the wide range of settlement values by managing complexity more effectively.

5. **Light Gradient Boosting (LightGBM) Regressor**
   - **Rationale**: Gradient boosting variant optimized for faster training and leaf-wise tree growth.
   - **Design**: Grows trees leaf-wise rather than level-wise for better accuracy with faster computation.
   - **Purpose**: Efficiently handles the large feature space created after encoding categorical variables.

### Deep Learning Approach

6. **Deep Neural Network (DNN)**
   - **Rationale**: Captures complex, potentially non-linear relationships between features and settlement values.
   - **Design**: Multiple dense layers with ELU activation, dropout for regularization, and early stopping.
   - **Purpose**: Learns hierarchical feature representations that may better model the complex factors affecting settlement amounts.

### Ensemble and Hybrid Approaches
**Things to Note: All the Models that has 2 tiers (Ensemble + DNN) will be using Out-Of-Fold Method to produce the data that feeds into the Meta-Learners (Most of the time the DNN) this action is delibrate to prevent model leakage where the model is trained on the test data as well leading to a misleadingly good model performance**

7. **Gradient Boosting Ensemble**
   - **Rationale**: Combines strengths of multiple tree-based algorithms.
   - **Design**: Stacks RandomForest, XGBoost, and LightGBM with GBR as meta-learner.
   - **Purpose**: Leverages diverse learning algorithms to capture different patterns in the insurance data.

8. **Gradient Boosting Ensemble - DNN**
   - **Rationale**: Combines the feature learning capabilities of neural networks with the predictive power of gradient boosting.
   - **Design**: Uses GBR to generate meta-features that are stacked with original features and fed into a DNN.
   - **Purpose**: Attempts to capture both tree-based splits and neural network representations of the data.

9. **Random Forest - DNN Regressor**
   - **Rationale**: Similar hybrid approach but using RandomForest as the base model.
   - **Design**: Uses out-of-fold predictions to create reliable meta-features for the DNN.
   - **Purpose**: Tries to leverage the strength of RandomForest in handling categorical variables and missing values alongside DNN's representation power.

10. **Gradient Boosting Ensemble - DNN Regressor**
    - **Rationale**: Enhanced ensemble with multiple base models and DNN meta-learner.
    - **Design**: Generate OOF predictions from multiple models to create rich meta-features.
    - **Purpose**: Creates a more sophisticated blend of tree-based and neural approaches for improved prediction accuracy.

### Specialized Approaches

11. **Range Specific Regressor (Classifier + Regressor)**
    - **Rationale**: Different settlement value ranges may follow different patterns.
    - **Design**: First classifies claims into value ranges, then applies specialized regressors for each range.
    - **Purpose**: Addresses the challenge that high-value claims have different predictive factors than low-value claims.

12. **Clustering + Regression Algorithm**
    - **Rationale**: Natural clusters in the data may indicate different claim types with different prediction patterns.
    - **Design**: Uses K-means to identify claim clusters, then builds specialized models for each cluster.
    - **Purpose**: Discovers natural groupings in claims data for more targeted prediction.

13. **Natural Segment Model**
    - **Rationale**: Decision tree breakpoints reveal natural data segments with distinct characteristics.
    - **Design**: Uses decision tree rules to segment data, then trains specialized models per segment.
    - **Purpose**: Exploits the domain-specific breakpoints in insurance data for more accurate predictions.

14. **Enhanced Ensemble Model**
    - **Rationale**: Sophisticated range-based ensemble with optimized thresholds.
    - **Design**: Creates separate ensembles for low, medium, and high value claims.
    - **Purpose**: Addresses the challenge that high-value claims require different modeling approaches than low-value claims.

15. **Hybrid Model with Error Correction**
    - **Rationale**: Even good models have systematic errors that can be learned and corrected.
    - **Design**: Combines multiple models with an additional error correction model.
    - **Purpose**: Systematically addresses prediction residuals to minimize overall error.

16. **Hybrid Model with Feature Selection**
    - **Rationale**: Not all features contribute equally to prediction accuracy.
    - **Design**: Uses correlation-based feature selection to focus on most predictive features.
    - **Purpose**: Reduces noise and focuses on the most relevant features for settlement prediction.

17. **Final Weighted Ensemble (Chosen Model)**
    - **Rationale**: Different models capture different aspects of the data.
    - **Design**: Weighted combination of the Simplified and Hybrid models with optimized weights.
    - **Purpose**: Achieves the best overall performance by leveraging strengths of multiple approaches.

## Performance Metrics Comparison

| Model | R² | RMSE | MAE | MAPE |
|-------|-----|------|-----|------|
| Decision Tree Regressor | 0.8402 | 347.10 | 186.05 | 16.63% |
| Random Forest Regressor | 0.9144 | 254.20 | 146.81 | 13.51% |
| Gradient Boosting Regressor | 0.9206 | 244.59 | 140.30 | 12.53% |
| XGBoost Regressor | 0.9261 | 235.89 | 133.67 | 11.97% |
| LightGBM Regressor | 0.9282 | 232.73 | 130.40 | 11.63% |
| Deep Neural Network | 0.9294 | 230.71 | 127.82 | 11.54% |
| Gradient Boosting Ensemble | 0.9301 | 229.41 | 127.59 | 11.41% |
| Gradient Boosting Ensemble - DNN | 0.9308 | 228.14 | 126.37 | 11.29% |
| Random Forest - DNN Regressor | 0.9321 | 226.12 | 124.76 | 11.18% |
| Range Specific Regressor | 0.9294 | 230.71 | 127.82 | 11.50% |
| Natural Segment Model | 0.9308 | 228.15 | 126.56 | 11.34% |
| Enhanced Ensemble Model | 0.9344 | 222.38 | 124.01 | 11.08% |
| Hybrid Model with Error Correction | 0.9350 | 221.45 | 123.48 | 11.04% |
| Hybrid Model with Feature Selection | 0.9353 | 221.09 | 123.27 | 11.00% |
| **Final Weighted Ensemble** | **0.9362** | **208.15** | **103.47** | **8.99%** |

## Why the Final Weighted Ensemble Was Selected

The Final Weighted Ensemble was chosen as the optimal model for several compelling reasons:

1. **Superior Performance Metrics**: 
   - Achieved the highest R², indicating it explains 93.62% of the variance in settlement values
   - Lowest RMSE showing it makes the smallest prediction errors overall
   - Lowest MAE and MAPE, demonstrating consistent accuracy across all claim value ranges

2. **Complementary Model Components**:
   - Combines the Simplified Model (with feature selection) and the Hybrid Model with complementary strengths
   - The Simplified Model excels at generalizing common patterns using only the most relevant features
   - The Hybrid Model captures complex interactions and handles special cases, particularly high-value claims

3. **Optimal Weight Distribution**:
   - The model found optimal weights (approximately 0.2 for Simplified and 0.8 for Hybrid) through a systematic grid search
   - This weight distribution suggests both models contribute significantly to the final prediction

4. **Handling of High-Value Claims**:
   - The ensemble particularly benefits from the Hybrid Model's specialized handling of high-value claims
   - High-value claims are often the most difficult to predict but have the highest financial impact on insurers

5. **Feature Efficiency**:
   - Leverages correlation-based feature selection to reduce dimensionality while maintaining predictive power
   - This makes the model more interpretable and computationally efficient

6. **Robustness to Outliers**:
   - Incorporates robust regression techniques (HuberRegressor for blending) that are less sensitive to outliers
   - This is particularly important for insurance claims where extreme values can occur

7. **Error Correction Mechanism**:
   - Incorporates systematic error correction that learns and addresses prediction residuals
   - This produces more calibrated predictions across the entire range of settlement values

**Things To Note** 
- The final_model in the Independant Deployment Section has slightly worse performance than that of the final_model above this is to be expected. Due to several factors including the Data Splitting Strategy, Weight Optimization Process, Modularity and Performance TradeOff. 

## Conclusion : This is the framework we will be using but for the model we will export the one with best performance that being the final_model (non-independant build)