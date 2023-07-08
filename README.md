# terumo-service-multiple-models

# Pipenv setup

pipenv is a popular package manager for Python that allows you to manage dependencies for your project in an isolated environment. Here are the basic steps for using pipenv:


bash
Copy code
1. Install pipenv using pip:

```bash
pip install pipenv
```

2, Creating a project and setting python version:
```bash
pipenv --python 3.10.4
```
2. Navigate to your project directory and run pipenv shell to activate a new virtual environment:
```bash
cd /path/to/your/project
pipenv shell
```
3. Create a new Pipfile and add a package to it:
```bash
pipenv install requests

# Dev dependencies
pipenv install --dev pytest
pipenv install --dev pytest-cov 
pipenv install --dev blue # PEP8 Formater
pipenv install --dev isort # import sorting
pipenv install --dev taskipy # automation scripting
pipenv install --dev safety

pipenv install --dev safety pytest pytest-cov blue isort taskipy
```

4. Alternatively, you can create a Pipfile manually and add packages to it using pipenv install:
```bash
touch Pipfile
pipenv install requests
```

5. To install packages listed in an existing Pipfile.lock, run pipenv sync:
```bash
pipenv sync
```

6. To uninstall a package, use pipenv uninstall:
```bash
pipenv uninstall requests
```

7. To exit the virtual environment, run exit:
```bash
exit
```

That's the basic workflow for using pipenv. It provides a simple and convenient way to manage dependencies for your Python projects.


# Taskpy
 In order to automate process we are using taskpy. You can check the tasks available using the  command. 
 ```bash
 task -l
 ```
 
 If you desire to create new tasks you shoud inset them on the "pyproject.toml"


# Install the module into your pipenv environment using pip:




Generate a lock file:

```bash
pipenv lock
```

Generate a requirements.txt file with only production dependencies:

```bash
pipenv requirements > requirements.txt
pipenv requirements --dev-only > requirements.txt
pipenv requirements --dev > dev-requirements.txt
```

pipenv install

python api/app.py