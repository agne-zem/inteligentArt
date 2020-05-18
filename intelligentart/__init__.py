from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_jsglue import JSGlue

app = Flask(__name__)

jsglue = JSGlue(app)

app.config['UPLOAD_FOLDER'] = './intelligentart/static/generated'
app.config['TEMP_FOLDER'] = './intelligentart/static/temp'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///intelligentart.db'

db = SQLAlchemy(app)

from intelligentart import routes