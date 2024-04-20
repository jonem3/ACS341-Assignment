# ACS341 Assignment - Spring 2024, Matthew Jones (Registration Num: 200326702)

Within the folder structure for this assignment, I have stored all the dependencies in order to run the project in a
requirements file as standard for python projects.

### Instructions for setup

**Please note:** The first time the assignment code is run, it will tune the Tensorflow Model hyperparameters, and save them to a folder within the project.
This will take
quite some time. After this, I would recommend re-running so all relevant information can be
seen on the same screen.

*For some of the following commands, `python3` may need to be replaced with `python` depending on your environment variables. Please verify that `python --version` is on at least version `3.10` if this is the case, as this project will not work in Python 2 and may not work on older versions of Python 3*

*The majority of these commands are there to keep a clean base install of python, if you are on a managed University computer or a computer where you do not have administrator privileges, skip to step 7*

1. Ensure Python 3 is installed and pip is up-to-date
2. Open a terminal or command prompt at the root folder of the project
3. Run `pip3 install virtualenv`
4. Create a new python virtual environment by running `python3 -m venv venv`
5. Activate the virtual environment using `source venv/bin/activate` on Linux or `venv\Scripts\activate.bat` on Windows
6. Your terminal environment should now show `(venv) $` to indicate successful activation 
7. Install all dependencies by running `pip3 install -r requirements.txt`
8. Run the `main.py` file by running `python3 main.py`

*Known Issues*
1. `Error: Can not perform a '--user' install. User site-packages are not visible in this virtualenv` FIX: Go to `pyvenv.cfg` and set `include-system-site-packages` to `true` and reactivate venv