# Semi-Automated Offside Detection System
- Digital Systems Project for final year Computer Science BSC at UWE.
- Written in Python and Javascript, using various computer vision libraries with Django and FastAPI.

## Setup
1.  Clone repository
2.  CD into repository
3.  Run `cp .env.example .env`
4.  Populate the env file:  
    a. Create an account on Roboflow.  
    b. Follow these instructions to get your Roboflow API key:
    https://docs.roboflow.com/api-reference/authentication
5.  Run `pip install -r requirements.txt`
5.  Run `cd app && python manage.py runserver`
6.  In another terminal window run `cd app/algorithm_api && uvicorn main:app --host 0.0.0.0 --port 8002 --reload`

## Accounts
### Create Account
1.  Navigate to `http://127.0.0.1:8000/admin/`
2.  Under 'AUTHENTICATION AND AUTHORIZATION' on the 'Users' row click 'Add'
3.  Fill out the 'Add user' form.

### Login
1.  Navigate to `http://127.0.0.1:8000`
2.  Click 'Login' in the 'Navigation' sidebar.
3.  Fill in user details.
