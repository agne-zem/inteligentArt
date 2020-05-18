from intelligentart import db


class Content(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    file_name = db.Column(db.String(120), unique=True, nullable=False)
    processing_status = db.Column(db.Integer, nullable=False)
    style = db.relationship('Style', backref='content', lazy=True)
    generated = db.relationship('Generated', backref='content', lazy=True)

    def __repr__(self):
        return f"Content('{self.id}', '{self.file_name}', " \
               f"'{self.processing_status}')"


class Style(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    file_name = db.Column(db.String(120), unique=True,nullable=False)
    fk_content_id = db.Column(db.Integer, db.ForeignKey('content.id'), nullable=False)

    def __repr__(self):
        return f"Style('{self.id}', '{self.file_name}', '{self.fk_content_id}')"


class Generated(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    file_name = db.Column(db.String(120), unique=True, nullable=False, default='default.png')
    rating = db.Column(db.Integer)
    type = db.Column(db.Integer, nullable=False, default=1)
    content_weight = db.Column(db.Integer, nullable=False)
    selected = db.Column(db.Boolean, nullable=False, default=False)
    custom = db.Column(db.Boolean, nullable=False, default=True)
    epochs = db.Column(db.Integer, nullable=False, default=2)
    steps = db.Column(db.Integer, nullable=False, default=20)
    fk_content_id = db.Column(db.Integer, db.ForeignKey('content.id'), nullable=False)

    def __repr__(self):
        return f"Generated('{self.id}', '{self.file_name}', '{self.rating}'," \
               f"'{self.type}', '{self.content_weight}', '{self.selected}', '{self.custom}', '{self.epochs}', " \
               f"'{self.steps}', '{self.fk_content_id}')"
