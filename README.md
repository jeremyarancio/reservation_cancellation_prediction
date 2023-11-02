# Vacasa assignement

Hey, I'm Jérémy, NLP engineer for 2 years and Research Engineer for 5 years before that.
I hope you're going to enjoy reading this project.
I had a ton of pleasure doing it, especially the engineering part.

To know more about me, check my [website](https://linktr.ee/jeremyarancio)! 

# How the project is structured

The way I tackled this project can be divided into 4 parts: 
* Exploratory data analysis
* Data preprocessing
* Model selection and training
* MLOps: reproducibility and deployment

But the repository is organized in a more engineer way. 

Let me walk you through my thought process then present you the enginneer part after.

## Thought process

### Exploratory Data Analysis

The dataset that was provided was extremely clean. Thus, it didn't need a lot of preprocessing.

One element I mainly focused on was the categorical features (months, etc...)
To avoid the [curse of dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality) in the case a number of unique categories high, I decided to go with the [TargetEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html) algorithm, and simple a [OrdinalEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html) algorithm otherwise.

All the exploratory was done in the following notebook: `notebooks/0_exploratory_data_analysis.ipynb`

### Data preprocessing

Based on the EDA, I wrote a Python script directly implemented into the MLOps part for reproducibility and later for inference. (I don't like notebooks for preprocessing data since it's hard to make it reproducible in a project)

Some columns associated with the target column were of course removed (`reservation_status` for example).
Otherwise, it would have biased the model.

*"The sky is blue. What color is the sky? You know what I mean...*

The script is located here: `ml_pipeline/pipeline/preprocess_step.py`
It acts as a `Step` easy to implement in `Pipelines`

### Model selection and training

I went with Gradient Boosting and got a `roc_auc` of **96%**, which is quite great. 

I'm surprise by this impressive results on first tries. 
I didn't have much time to dig deeper into it, so I'm open to discuss it.

The train script is located at: `ml_pipeline/pipeline/train_step.py`

### MLOps: reproducibility and deployment

My favorite part! 

MLFlow was used to track experiments and as a model registry.
MLFlow API was used to track metrics, log artifacts and register a model in the model registry if the metric was better compared to the previous registered model version.

I kept MLFlow data in the repo (`mlflow/`) for the sake sharing the project with you, but deploying it on a docker container on EC2 or Heroku is the way. 

Data was kept in the repository for more simplicity, but using DVC would have been better in this case (small dataset).

Regarding the reproducibility, I built a `Pipeline` composed of `Steps`. Each `Step` has its own purpose and connect with data and MLflow. 

You can change the pararemeters of the model, the criteria to register a new model or change the metric pretty easily.

I invite you to explore the `ml_pipeline` to get a better grasp.

`train.py` is the main script to preprocess, train and register the model based on conditions
`inference.py` is the main script to preprocess the new batch and predict.
*I still have to add the TargetEncoder and OrdinalEncoder to the artifact for inference*

```bash
├── ml_pipeline
    ├── pipeline
    │   │   ├── condition_step.py
    │   │   ├── config.py
    │   │   ├── inference_step.py
    │   │   ├── pipeline.py
    │   │   ├── preprocess_step.py
    │   │   ├── train_step.py
    │   │   └── utils
    │   │       └── step.py
│   ├── inference.py
│   └── train.py
```

# Try it by yourself

## Set up your environment
Create a new environment (python=3.9) and run:

```bash
pip install -r requirements.txt
```

To ensure the python scripts run correctly, don't forget to add the repo path the PYTHONPATH.
You can do by adding to `.bashrc` the following:

```bash
export PYTHONPATH=path/to/the/repo
```

## Start MLflow

To start Mlflow UI, run:

```bash
mlflow server --backend-store-uri mlflow/ --artifacts-destination mlflow/ --port 8000
```

You'll normally see the previous experiments and the model stored in the model registry.

## Run the training

```bash
python ml_pipeline/train.py
```

## Run Inference
*Work in Progress*


...

I hope you're going to enjoy it!


