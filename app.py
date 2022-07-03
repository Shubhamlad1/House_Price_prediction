from distutils.debug import DEBUG
import sys
from flask import Flask 
from housing.logger import logging
from housing.exception import HousingException

app=Flask(__name__)

@app.route("/",methods=["GET", "POST"])
def index():
    try:
        raise Exception("rainsng a custom exception for test")
    except Exception as e:
        housing= HousingException(e, sys)
        logging.info(housing.error_message)
        logging.info("Logging module started")
    
    return "CI/CD Pipeline Created"
    

if __name__=="__main__":
    app.run(debug=True)

