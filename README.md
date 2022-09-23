Introduction: 

House Price prediction is an important topic which helps in understanding the current prices of the properties according to the locations, room, carpet area available ect. Considering predictions, it helps in better decision-making. Today we have large amount of data available from the various sources and we can make better utilization of data by using different Machine Learning Algorithms.

In this project we have established complete machine learning pipeline. VS Code IDE is being used for project development, Git Bash and Git tool is used for version control, Docker Images are used for establishing CICD pipeline and the complete project is deployed on Heroku platform.

Pandas, NumPy, Scikit-learn, Flask, dill, evidently are some of the important libraries being used to develop machine learning pipeline. This this project we have used dynamic approach to perform EDA, Feature Engineering, and Model Building. As data Is changing frequently, keeping our model up to date is important. In this case dynamic approach in the programming language helps us to trigger the pipeline and get the results accordingly. Also Logging and exception handling is being used in the project where it is required.  

Important Files:

Config.yaml :

This file contains details required to establish project configuration. To create the folders to save input files and artifact file we are using the values in the canfig.yaml file.

Schema.yaml:

This file contain information related to the data such as data types of the columns, numerical columns, categorical columns and target columns.

Model.yaml:

This file contains the information related to Algorithms used for model building.

Steps Followed:

Data Ingestion:

•	Data is downloaded from the URL.\n
•	split into train test files and saved.

Data Validation:

•	Checking the data types of columns are altering it if required.
•	Checking Data Drifts between new and old data. (Evidently library is used)

Data Transformation:

•	Checking and handling null values.
•	Data standardization using StandardScaler and OneHotEncoder libraries from Scikit-learn.

Model Training and Evaluation:
•	LinearRegression and RandomForestRegression is used with GridSeachCV.
•	Model which is giving the best and generalized accuracy are selected and being used for predicting the values. 


<img width="913" alt="Project Structure" src="https://user-images.githubusercontent.com/96531123/191944094-270cf36b-3980-4582-980c-c4253650f412.png">

