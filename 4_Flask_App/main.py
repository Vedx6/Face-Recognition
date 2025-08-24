<<<<<<< HEAD
from flask import Flask # type: ignore
from app import views

app = Flask(__name__) # webserver gateway interphase (WSGI)

app.add_url_rule(rule='/',endpoint='home',view_func=views.index)
app.add_url_rule(rule='/app/',endpoint='app',view_func=views.app)
app.add_url_rule(rule='/app/gender/',
                 endpoint='gender',
                 view_func=views.genderapp,
                 methods=['GET','POST'])

if __name__ == "__main__":
=======
from flask import Flask
from app import views

app = Flask(__name__) # webserver gateway interphase (WSGI)

app.add_url_rule(rule='/',endpoint='home',view_func=views.index)
app.add_url_rule(rule='/app/',endpoint='app',view_func=views.app)
app.add_url_rule(rule='/app/gender/',
                 endpoint='gender',
                 view_func=views.genderapp,
                 methods=['GET','POST'])

if __name__ == "__main__":
>>>>>>> 795f1e101b30f90cf3dceed5d742bb8375e6f712
    app.run(debug=True)