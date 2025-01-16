# ASSIGNMENT 3
## CONTRIBUTORS
[Mai Chieu Thuy - s3877746](https://github.com/thuyiswater)

[Le Cam Tu - s3915195](https://github.com/hydl19903)

[Dinh Ngoc Minh - s3925113](https://github.com/dinhminh0307)

[Nguyen Mau Bach - s3926937](https://github.com/Helixu38)

## HOW TO RUN


### 2.2. Analysis and Hypothesis Proposal
#### 2.2.1. Regression problem
 
#### 2.2.2. Clustering problem
 
#### 2.2.3. Classification problem
##### 1. Visitor Engagement
- **Observation**: Scatterplots of `Administrative_Duration`, `Informational_Duration`, and `ProductRelated_Duration` against `Revenue` often show higher densities of sessions with greater durations linked to purchases (`Revenue = True`).
  - `PageValues` seems to have a strong positive correlation with `Revenue`. Higher page values indicate a higher likelihood of purchases, aligning with the hypothesis.

- **Hypothesis**: Higher engagement leads to higher purchase probabilities.
  - Higher values for `Administrative_Duration`, `Informational_Duration`, and `ProductRelated_Duration` might indicate greater visitor interest, resulting in purchases.
  - A higher `PageValues` score is likely to correlate positively with purchases.

##### 2. Bounce and Exit Rates
- **Observation**: From scatterplots of `BounceRates` and `ExitRates` against `Revenue`, sessions with higher bounce rates (`BounceRates`) and exit rates (`ExitRates`) generally correspond to no purchases (`Revenue = False`), which supports the hypothesis of poor user experience reducing purchase likelihood.

- **Hypothesis**: Poor user experience decreases purchase likelihood.
  - Higher `BounceRates` and `ExitRates` might indicate user dissatisfaction, leading to fewer purchases.

##### 3. Time Factors
- **Observation**: Visualizations of `SpecialDay` against `Revenue` show an increase in purchases as the proximity to a special day increases. Similarly, analysis of `Month` against `Revenue` highlights seasonal trends, with months like November showing higher purchase probabilities, likely due to shopping holidays like Black Friday.

- **Hypothesis**: Shopping behavior depends on timing.
  - Visits closer to a `SpecialDay` (e.g., Black Friday or holidays) might have a higher likelihood of purchases.
  - Certain months (`Month`) might reflect seasonal shopping trends, influencing purchase behavior.

##### 4. User Types
- **Observation**: From categorical plots of `VisitorType` and `Revenue`, returning visitors have a noticeably higher likelihood of generating revenue compared to new visitors. This observation supports the hypothesis that returning visitors are more likely to purchase.

- **Hypothesis**: Returning visitors are more likely to purchase.
  - Returning visitors (`VisitorType = Returning_Visitor`) might have a higher likelihood of making a purchase compared to new visitors (`VisitorType = New_Visitor`).

##### 5. Technical Features
- **Observation**: Bar charts and heatmaps for `OperatingSystems`, `Browser`, and `TrafficType` reveal varying purchase probabilities across categories. For instance, some browsers or traffic sources have a stronger association with purchases, supporting the hypothesis that technical accessibility impacts purchases.

- **Hypothesis**: Technical accessibility impacts purchases.
  - Different `OperatingSystems`, `Browser`, and `TrafficType` values might influence the likelihood of purchases based on usability or accessibility.

##### 6. Weekend Influence
- **Observation**: A categorical plot of `Weekend` against `Revenue` shows slight differences in purchase likelihood between weekend and weekday sessions. This observation suggests some behavioral differences in shopping patterns based on the day of the week.
- **Hypothesis**: Shopping behavior differs on weekends.
  - Visits during weekends (`Weekend = True`) might have different purchase rates compared to weekday visits.

---
