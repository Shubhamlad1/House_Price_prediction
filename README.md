# Machine-Learning-Project-1
This is first ML Project

Creating conda environment
'''
conda create -p project_1_env python==3.9 -y
'''

After Creating Environment activate the created Environment
'''
activate project_1_env/ 
'''

To create the CI/CD pipeline following information is required:
Heroku_Email: shubhamlad33@gmail.com
Heroku_API_Key: <>
Heroku_APP_Name: machine-learning-project-01

Build Docker Image
'''
docker build -t <Image-name>:<tagname>
'''
> Note: Image Name should always be in small letter

To list the docker image:
'''
docker images
'''

Run Docker images:
'''
docker run -p 5000:5000 -e PORT=5000 c7382c34a154
'''

To check running containers in the docker
'''
docker ps
'''

to stop runnig docker file:
'''
docker stop <Container ID>
'''
> Note: Container Id Can be found after running docker ps command

Docker Command:
'''
CMD gunicorn --workers=4 --bind 0.0.0.0:$PORT app:app
'''
>Note: 1. $PORT is we are allowing heroku to assign a port for our app to run.
       2. app:app = module-file-of-flask:object-name-of-flask

