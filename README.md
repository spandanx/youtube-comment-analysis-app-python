# Youtube Comment Analyzer Application

This a web application that can combine multiple youtube content and give a summarized view of comments and possible question and answers.

## Features
 - Get Summary of youtube comments and description
 - Get possible question and answers
 - Login with Oauth authentication

<details>
   <summary>Application Snap Shots</summary>
  <p>LoginPage</p>
  <img src="https://github.com/user-attachments/assets/eedfea3b-9f27-40c8-a8ae-658d2b34d248" width=50% height=50%>
  <p>RegistrationPage</p>
  <img src="https://github.com/user-attachments/assets/26fd57ee-51b4-442c-9aed-cf5e54c7bd6c" width=50% height=50%>
  <p>HomePage</p>
  <img src="https://github.com/user-attachments/assets/0f0a5bc4-ae9f-4b80-8f7e-8d49bb2c0028" width=50% height=50%>
  <p>Model Selection</p>
  <img src="https://github.com/user-attachments/assets/7353dcc3-80ce-444e-aa6c-715e8213c87c" width=50% height=50%>
  <p>VideoList</p>
  <img src="https://github.com/user-attachments/assets/1a66b488-863b-4ae3-8c49-03e6d58d886c" width=50% height=50%>
  <p>SummaryList</p>
  <img src="https://github.com/user-attachments/assets/b8a0cf26-91be-4e3c-8887-4ae2b6d3d00e" width=50% height=50%>
  <p>QuestionList</p>
  <img src="https://github.com/user-attachments/assets/be73f57e-fd68-49bf-8637-fbb140f24a45" width=50% height=50%>

</details>
	
## Technologies used

Frontend - `React.js`

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
<p>Open miniconda console. Run the below commands </p>

```
conda create -n env-name python=3.10
conda activate env-name
```

#### Install required packages
```
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

#### Go to Python shell
```
import nltk
nltk.download()
```
#### Run the python application
```
python -m uvicorn main:app --reload --env-file path-to-env-file/custom_env_data.env
```
The environment file should contain the youtube api key, the text encoding etc.

## Architecture
### High Level Design Diagram


### Activity Diagram
<img src="https://github.com/user-attachments/assets/22e0bede-a581-4ac8-ae46-b4aebd8dd390" width=35% height=35%>

### Functional Diagram
![FunctionalDiagram_Extract_Text_Youtube_Comment_Analyzer](https://github.com/user-attachments/assets/de183f3d-4b38-4142-bdda-5e7a469ed12e)
![FunctionalDiagram_Summary_Youtube_Comment_Analyzer](https://github.com/user-attachments/assets/ebe398fd-e50c-44ab-b7fc-ae4417717f03)
![FunctionalDiagram_QA_Youtube_Comment_Analyzer](https://github.com/user-attachments/assets/53ed1a44-f129-4cb1-8ef6-c64b4c633645)


## Youtube link


