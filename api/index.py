from flask import Flask
import sys, os

# Add your 4_Flask_App folder to the Python path so imports work
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '4_Flask_App'))

from app import views

app = Flask(__name__)

app.add_url_rule(rule='/', endpoint='home', view_func=views.index)
app.add_url_rule(rule='/app/', endpoint='app', view_func=views.app)
app.add_url_rule(
    rule='/gender/',
    endpoint='gender',
    view_func=views.genderapp,
    methods=['GET', 'POST']
)

# Vercel looks for a function that takes (environ, start_response)
def handler(environ, start_response):
    return app.wsgi_app(environ, start_response)
