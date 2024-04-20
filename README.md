# ACS341 Assignment - Spring 2024, Matthew Jones (Registration Num: 200326702)

Within the folder structure for this assignment, I have stored all the dependencies in order to run the project in a
requirements file as standard for python projects.

### Instructions for setup

**Please note:** The first time the assignment code is run, it will tune the Tensorflow Model hyperparameters, and save them to a folder within the project.
This will take
quite some time. After this, I would recommend re-running so all relevant information can be
seen on the same screen.

*For some of the following commands, `python3` may need to be replaced with `python` depending on your environment variables*

1. Ensure Python 3 is installed and pip is up-to-date
2. Open a terminal or command prompt at the root folder of the project
3. Run `pip3 install virtualenv`
4. Create a new python virtual environment by running `python3 -m venv venv`
5. Activate the virtual environment using `source venv/bin/activate` on Linux or `venv\Scripts\activate.bat` on Windows
6. Your terminal environment should now show `(venv) $` to indicate successful activation 
7. Install all dependencies by running `pip3 install -r requirements.txt`
8. Run the `main.py` file by running `python3 main.py`