# Youtube Comment Analyzer Application

This a web application that can combine multiple youtube content and give a summarized view of comments and possible question and answers.

## Features
 - Get Summary of youtube comments and description
 - Get possible question and answers
 - Login with Oauth authentication

<details>
   <summary>Application Snap Shots</summary>
  <p>LoginPage</p>
  <img src="https://github.com/user-attachments/assets/eedfea3b-9f27-40c8-a8ae-658d2b34d248" width=25% height=25%>
  <p>RegistrationPage</p>
  <img src="https://github.com/user-attachments/assets/26fd57ee-51b4-442c-9aed-cf5e54c7bd6c" width=25% height=25%>
  <p>HomePage</p>
  <img src="https://github.com/user-attachments/assets/0f0a5bc4-ae9f-4b80-8f7e-8d49bb2c0028" width=25% height=25%>
  <p>Model Selection</p>
  <img src="https://github.com/user-attachments/assets/7353dcc3-80ce-444e-aa6c-715e8213c87c" width=25% height=25%>
  <p>VideoList</p>
  <img src="https://github.com/user-attachments/assets/1a66b488-863b-4ae3-8c49-03e6d58d886c" width=25% height=25%>
  <p>SummaryList</p>
  <img src="https://github.com/user-attachments/assets/b8a0cf26-91be-4e3c-8887-4ae2b6d3d00e" width=25% height=25%>
  <p>QuestionList</p>
  <img src="https://github.com/user-attachments/assets/be73f57e-fd68-49bf-8637-fbb140f24a45" width=25% height=25%>

</details>
	
## Technologies used

Frontend - React.js

Backend - `FastAPI`, `OAuth2 Security`, `Python`, `KMeans Clustering`, `LSTM`, `Huggingface Summarizer`

## Steps to run locally



### Step 1. Clone the repositories
#### Backend: https://github.com/spandanx/youtube-comment-analysis-app-python
#### Frontend: https://github.com/spandanx/youtube-comment-analysis-reactjs


### Step 2. Install required softwares

`Node.js`
`Miniconda`

### Step 3. Prepare backend

#### Create new environment
<p> Open miniconda console </p>
<p> Run the below commands </p>
<p> conda create -n env-name python=3.10 </p>
<p> conda activate env-name </p>

#### Install required packages
<p> python -m pip install -r requirements.txt </p>
<p> python -m spacy download en_core_web_sm </p>

#### Go to Python shell
<p> import nltk </p>
<p> nltk.download() </p>

#### Run the python application
<p>python -m uvicorn main:app --reload --env-file path-to-env-file/custom_env_data.env</p>
<p>The environment file should contain the youtube api key, the text encoding etc.</p>

## Architecture
### High Level Design Diagram


### Activity Diagram
<img src="https://github.com/user-attachments/assets/22e0bede-a581-4ac8-ae46-b4aebd8dd390" width=35% height=35%>

### Functional Diagram
<p></p>
<img src="https://github.com/user-attachments/assets/53ed1a44-f129-4cb1-8ef6-c64b4c633645" width=50% height=50%>
<p></p>
<img src="https://github.com/user-attachments/assets/1e60529e-c0d5-4de5-adfb-a0d04ab05007" width=50% height=50%>
<p></p>
<img src="https://github.com/user-attachments/assets/52e526ef-2864-4f4e-ad15-325f45682b3c" width=50% height=50%>


## Youtube link


