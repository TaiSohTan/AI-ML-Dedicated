# Exploratory Data Analysis Findings

## Dataset Overview

### Data Structure
- Dataset contains insurance claim records with both categorical and numerical features
- Missing data pattern shows 3,545 rows (approximately 71%) have complete data
- 305 instances have 1 missing feature, 328 have 2 missing features, 324 have 3 missing features, 311 have 4 missing features and 187 have 5 missing features. No instances have more than 5 missing features.
- Some features show correlated missing patterns in the heatmap visualization
- Data spans multiple accident types, injury classifications, and settlement components

### Variables of Interest
- Target variable (`SettlementValue`) represents the final payout amount
- Temporal data includes `Accident Date` and `Claim Date`, showing claim filing patterns
- Injury-related features include `Minor_Psychological_Injury`, `Whiplash`, `Dominant injury`, and `Injury_Prognosis`
- Financial components include `SpecialAssetDamage`, `GeneralRest`, `SpecialTherapy`, and `GeneralFixed`
- Contextual information includes `Vehicle Type`, `AccidentType`, and demographic data

## Settlement Value Distribution

### Statistical Summary
- Mean settlement: Approximately £1,218.01
- Median settlement: Lower than mean, indicating right-skewed distribution
- Minimum value: Around £240 (representing minimum payout)
- Maximum value: Around £7,862.90 (representing extreme cases)
- Standard deviation shows considerable variation in settlement amounts

### Distribution Characteristics
- Strong positive skew with most claims clustering at lower values
- Long right tail containing high-value outliers
- The 95% confidence interval (between 2.5th and 97.5th percentiles) shows the typical range where most claims fall
- The distribution suggests potential for logarithmic transformation in modeling

## Key Relationships

### Injury Types and Combinations

#### Derived InjuryCombo Feature
- "Psych Only" cases: 1,870 claims with average settlement of approximately £1,220
- "Whiplash Only" cases: 818 claims with lower average settlement of approximately £1,029
- "Both" injury types: 1,213 claims with highest average settlement of approximately £1,304
- "None" (no documented injury): 1,099 claims with settlements averaging around £1,133
- Distribution shows psychological injuries appear more frequently than whiplash

#### Injury Prognosis Impact
- Longer recovery prognosis periods (especially 12+ months) strongly correlate with higher settlements
- The relationship appears somewhat linear between prognosis duration and settlement amount
- The strongest effect is visible in high-value claims analysis

### Accident Types and Severity

#### Rear-end Accidents
- Represent the most common accident type in the dataset
- Show distinct settlement patterns compared to other accident types
- Settlement distribution for rear-end accidents shows characteristic clustering

#### Non-rear-end Accidents
- Comprise approximately 30-40% of total accidents
- Show greater variability in settlement amounts
- Certain types (potentially those involving intersections or higher speeds) correlate with higher settlements
- The boxplot visualization reveals several categories with significantly higher median settlements

## High-Value Claims Analysis

### 95th Percentile Claims
- Represent approximately 5% of all claims but a disproportionate share of total payout value
- Show distinct characteristic patterns compared to average claims
- Have higher rates of certain accident types and injury combinations

### Distinguishing Factors
- Certain accident types are overrepresented by 15-30% compared to their occurrence in all claims
- "Both" injury combination (psychological + whiplash) appears more frequently
- Longer injury prognosis periods (particularly 12+ months) are substantially overrepresented
- Specific vehicle types show higher likelihood of resulting in high-value claims

## Temporal Patterns

### Accident and Claim Date Analysis
- Clear seasonal patterns visible in accident frequency
- Consistent lag between accident date and claim filing date
- Monthly trend analysis shows potential yearly cyclical patterns
- Possible correlation between certain months and higher settlement values

## Numerical Feature Relationships

### Correlation Analysis
- Strong positive correlations between certain payment components and total settlement
- `GeneralRest` and `SpecialTherapy` show particularly strong influence on final settlement amount
- `Driver Age` shows weak but potentially meaningful correlation with settlement values
- Boxplots for numerical features reveal several that contain significant outliers

## Notable Plots

#### 1. Settlement Value Distribution (Histogram with KDE)
   
   ![Settlement Value Distribution (Histogram with KDE)](/images/SettlementValueHistKDE.png)

#### 2. Settlement Value by Injury Combination (Enhanced Boxplots)

   ![Settlement Value Distribution by Injury Combination](/images/InjuryComboBoxPlot.png)

#### 3. Missing Data Matrix 
   
   ![Missing Value Randomness Research](/images/MsnoHeatmap.png)

#### 4. Accident Type Analysis 

   ![Accident Type Settlement Value Analysis](/images/AccidentTypeSettlementValueBoxPlot.png)

#### 5. Mean Settlement Value for Non-Rear End Accident
   
   ![Mean Settlement Value for Non-Rear End Accident Type](/images/MeanSettlementValueByNonRearEnd.png)