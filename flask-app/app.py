import json
from flask import Flask, request, jsonify
import os
import subprocess

app = Flask(__name__)
@app.route("/", methods=['GET']) # give your path and methods you want, GET is default

def your_function_name():
    # you will have request object available here for data
    # return your response
    unzip_command = "unzip site1.zip"
    unzip_result = subprocess.check_output(unzip_command, shell=True)
   
    venv_command = "sh ./virtualenv/set_env.sh"
    venv_result = subprocess.check_output(venv_command, shell=True)

#    venv_activate_command = "source nvflare/bin/activate"
#    venv_activate_result = subprocess.check_output(venv_activate_command, shell=True)

    pip3_install_command = "pip3 install -r ./virtualenv/min-requirements.txt"
    pip3_install_result = subprocess.check_output(pip3_install_command, shell=True)

    pip3_install_command2 = "pip3 install -r ./virtualenv/plot-requirements.txt"
    pip3_install_result2 = subprocess.check_output(pip3_install_command2, shell=True)
   
   
    client_start_command = "sh ./startup/start.sh"
    client_start_result = subprocess.check_output(client_start_command, shell=True)
   
    return ("Client setup complete")
   
if __name__ == '__main__':
    app.debug = True
    app.run()
