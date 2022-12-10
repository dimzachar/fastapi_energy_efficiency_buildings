## Project of the week FastAPI for Energy Efficiency of Buildings

I update here my [Midterm Project](https://github.com/dimzachar/mlzoomcamp_projects/tree/master/00-midterm_project) to use FastAPI.

Repo contains the following:

* `README.md` with
  * Instructions on how to run the project
* Data `ENB2012_data.csv`
* `notebook.ipynb` a Jupyter Notebook with the data analysis and models
* Script `train.py` (updated to use Pipeline)
* Script `predict.py` (uses FastAPI with Pydantic)
* Json files `test.json` and `test2.json` to test the service. Change them to produce another prediction.
* Files with dependencies
  * `env_project.yml` conda environment (optional)


## Local deployment

All development was done on Windows with conda.

You can either recreate my environment by
```bash
conda env create -f env_project.yml
conda activate project
```

or do it on your own environment.

1. Download repo
```bash
git clone https://github.com/dimzachar/mlzoomcamp_projects.git
```

2. For the virtual environment, I utilized pipenv. If you want to use the same venv as me, install pipenv and dependencies, navigate to the folder with the given files:
```bash
cd 00-midterm_project
pip install pipenv
pipenv install numpy pandas seaborn bentoml tqdm jupyter scikit-learn==1.1.3 xgboost==1.7.1 pydantic==1.10.2
```

3. Enter shell. To open the `notebook.ipynb` and see all the models

```bash
pipenv shell
pipenv run jupyter notebook
```

For the following you need to run train.py
```bash
pipenv run python train.py
```

4. Then, get the service running on [localhost](http://localhost:8000)

```bash
pipenv run uvicorn predict:app --reload
```

and test it with the data in the `test.json` and `test2.json` on /docs page.

## Further development

Might try to check if async with await work here like in the BentoML case.


**Connect with me:**

<p align="center">
  <a href="https://www.linkedin.com/in/zacharenakis/" target="blank"><img align="center" src="https://cdn-icons-png.flaticon.com/512/174/174857.png" height="30" width="30" /></a>
  <a href="https://github.com/dimzachar" target="blank"><img align="center" src="https://cdn-icons-png.flaticon.com/512/25/25231.png" height="30" width="30" /></a>

  
</p>
           
