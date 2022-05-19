# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

- The model intends to predict whether US workers have income greater than 50k dollars or not
- The dataset contains more than 30 thousands registers about workers.
- Among the features available to predict person's salary, we could mention education, race, sex and occupation
- The algorithm used to predict income was GradientBoostingClassifier
- DVC was used as data versioning and controling
- The data was stored in S3 bucket

## Intended Use

- The use of this model is only useful for class exercise purpose once it was not optimazed to be used as an income forecaster guidance.
- Predict whether US worker makes more than 50K thousands.
- Automate Ml models based on the concept of CI/CD
- As RESTful API

## Training Data

- 80% of the dataset was set as training data.
- Data wrangling performed to avoid any model inconsistence.

## Evaluation Data

- 20% of the dataset has been used as test set.

## Metrics

_Please include the metrics used and your model's performance on those metrics._

- 2022-05-19 14:22:22,251 Precision: 0.7581027667984189
- 2022-05-19 14:22:22,251 Recall: 0.6159280667951188
- 2022-05-19 14:22:22,251 fbeta: 0.679659815733522

## Ethical Considerations

- This model should not be used to take any public or private decision regarding this matter as it has not been conceived to this purpose.

## Caveats and Recommendations

- There are many ways to improve this model, for example, once one consider to perform Hyper-parameters optimization.
