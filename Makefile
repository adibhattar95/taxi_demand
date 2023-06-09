.PHONE: requirements features

#Generate a requirements.txt from pyproject.toml if you work with Poetry
requirements:
		python -m pip install --upgrade pip
		pip-compile -o requirements.txt pyproject.toml --resolver=backtracking

#Run the feature pipeline
features:
		poetry run python scripts/feature_pipeline.py

frontend:
		poetry run streamlit run src/frontend.py