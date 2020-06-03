from intelligentart import db


# model class saving for content images to database
class Content(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    file_name = db.Column(db.String(120), unique=True, nullable=False)
    processing_status = db.Column(db.Integer, nullable=False)
    style = db.relationship('Style', backref='content', lazy=True)
    generated = db.relationship('Generated', backref='content', lazy=True)

    # for printing
    def __repr__(self):
        return f"Content('{self.id}', '{self.file_name}', " \
               f"'{self.processing_status}')"


# model class for saving style images to database
class Style(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    file_name = db.Column(db.String(120), unique=True, nullable=False)
    used = db.Column(db.Boolean, nullable=False, default=False)
    fk_content_id = db.Column(db.Integer, db.ForeignKey('content.id'), nullable=False)

    # for printing
    def __repr__(self):
        return f"Style('{self.id}','{self.used}', '{self.file_name}', '{self.fk_content_id}')"


# model class for saving generated images to database
class Generated(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    file_name = db.Column(db.String(120), unique=True, nullable=False, default='default.png')
    rating = db.Column(db.Integer)
    type = db.Column(db.Integer, nullable=False, default=1)
    content_weight = db.Column(db.Integer, nullable=False)
    selected = db.Column(db.Boolean, nullable=False, default=False)
    configuration_type = db.Column(db.Integer, nullable=False, default=1)
    epochs = db.Column(db.Integer, nullable=False, default=2)
    steps = db.Column(db.Integer, nullable=False, default=20)
    fk_content_id = db.Column(db.Integer, db.ForeignKey('content.id'), nullable=False)

    # for printing
    def __repr__(self):
        return f"Generated('{self.id}', '{self.file_name}', '{self.rating}'," \
               f"'{self.type}', '{self.content_weight}', '{self.selected}', '{self.configuration_type}', '{self.epochs}', " \
               f"'{self.steps}', '{self.fk_content_id}')"
