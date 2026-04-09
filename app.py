import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from flask import Flask, flash, redirect, render_template, request, url_for
from flask_login import (LoginManager, UserMixin, current_user, login_required,
                         login_user, logout_user)
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import check_password_hash, generate_password_hash

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'change-me-in-production')
app.config['SQLALCHEMY_DATABASE_URI'] = (
    os.environ.get('DATABASE_URL') or
    f"sqlite:///{os.path.join(BASE_DIR, 'coursepredictor.db')}"
)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access the portal.'
login_manager.login_message_category = 'warning'

# ---------------------------------------------------------------------------
# Database models
# ---------------------------------------------------------------------------

class User(UserMixin, db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    predictions = db.relationship('Prediction', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class Prediction(db.Model):
    __tablename__ = 'predictions'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    strand = db.Column(db.Integer, nullable=False)
    tier = db.Column(db.Integer, nullable=False)
    shs_general_average = db.Column(db.Float, nullable=False)
    language = db.Column(db.Float, nullable=False)
    science = db.Column(db.Float, nullable=False)
    general_knowledge = db.Column(db.Float, nullable=False)
    math = db.Column(db.Float, nullable=False)
    total = db.Column(db.Float, nullable=False)
    top1_course = db.Column(db.String(120))
    top1_confidence = db.Column(db.Float)
    top2_course = db.Column(db.String(120))
    top2_confidence = db.Column(db.Float)
    top3_course = db.Column(db.String(120))
    top3_confidence = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


# ---------------------------------------------------------------------------
# Domain mappings (kept in sync with the trained model)
# ---------------------------------------------------------------------------
STRAND_MAPPING = {
    1: 'ABM',
    2: 'ICT',
    3: 'HUMSS',
    4: 'GAS',
    5: 'STEM',
    6: 'FBS',
    7: 'BPP',
    8: 'TVL',
    9: 'EIM',
    10: 'HE',
    11: 'PEE-BACC',
    12: 'SMAW',
    14: 'HUMMS',
    15: 'Caregiving',
}

COURSE_MAPPING = {
    1: 'Bachelor of Library and Information Science',
    2: 'Bachelor of Science in Information Technology',
    3: 'Bachelor of Science in Business Administration',
    4: 'Bachelor of Public Administration',
    5: 'Bachelor of Elementary Education',
    6: 'Bachelor of Science in Tourism Management',
    7: 'Bachelor of Science in Computer Science',
    8: 'Bachelor of Science in Entrepreneurship',
    9: 'Bachelor of Science Elementary Education',
}

TIER_MAPPING = {
    1: 'Tier 1',
    2: 'Tier 2',
    3: 'Tier 3',
}

# ---------------------------------------------------------------------------
# ML pipeline – loaded once at start-up
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(BASE_DIR, 'log_model.pkl')

try:
    pipeline = joblib.load(MODEL_PATH)
except FileNotFoundError:
    pipeline = None


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _predict(strand, science, math, language, total, shs_general_average,
             tier, general_knowledge):
    """Return a list of top-3 prediction dicts or raise on error."""
    if pipeline is None:
        raise RuntimeError("Model file not found. Run create_model.py first.")

    features = pd.DataFrame({
        'LANGUAGE': [language],
        'SCIENCE': [science],
        'GENERAL_KNOWLEDGE': [general_knowledge],
        'MATH': [math],
        'TOTAL': [total],
        'SHS_GENERAL_AVERAGE': [shs_general_average],
        'SHS_STRAND': [strand],
        'TIER': [tier],
    })

    predicted_proba = pipeline.predict_proba(features)[0]
    top3_indices = np.argsort(predicted_proba)[-3:][::-1]
    top3_classes = pipeline.classes_[top3_indices]
    top3_scores = predicted_proba[top3_indices]

    return [
        {
            'course': COURSE_MAPPING.get(int(c), f"Course {c}"),
            'confidence': f"{s * 100:.1f}%",
            'confidence_raw': float(s),
        }
        for c, s in zip(top3_classes, top3_scores)
    ]


# ---------------------------------------------------------------------------
# Authentication routes
# ---------------------------------------------------------------------------

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')

        errors = []
        if not username or len(username) < 3:
            errors.append('Username must be at least 3 characters long.')
        if not email or '@' not in email:
            errors.append('A valid email address is required.')
        if len(password) < 6:
            errors.append('Password must be at least 6 characters long.')
        if password != confirm_password:
            errors.append('Passwords do not match.')
        if User.query.filter_by(username=username).first():
            errors.append('Username is already taken.')
        if User.query.filter_by(email=email).first():
            errors.append('Email is already registered.')

        if errors:
            for err in errors:
                flash(err, 'danger')
        else:
            user = User(username=username, email=email)
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        remember = bool(request.form.get('remember'))

        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user, remember=remember)
            next_page = request.args.get('next')
            flash(f'Welcome back, {user.username}!', 'success')
            return redirect(next_page or url_for('home'))
        else:
            flash('Invalid username or password.', 'danger')

    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))


# ---------------------------------------------------------------------------
# Prediction route
# ---------------------------------------------------------------------------

@app.route('/', methods=['GET', 'POST'])
@login_required
def home():
    predictions = None

    if request.method == 'POST':
        try:
            strand = int(request.form['STRAND'])
            science = float(request.form['SCIENCE'])
            math = float(request.form['MATH'])
            language = float(request.form['LANGUAGE'])
            total = float(request.form['TOTAL'])
            shs_general_average = float(request.form['SHS_GENERAL_AVERAGE'])
            tier = int(request.form['TIER'])
            general_knowledge = float(request.form['GENERAL_KNOWLEDGE'])

            predictions = _predict(strand, science, math, language, total,
                                   shs_general_average, tier, general_knowledge)

            # Persist the prediction
            pred_record = Prediction(
                user_id=current_user.id,
                strand=strand,
                tier=tier,
                shs_general_average=shs_general_average,
                language=language,
                science=science,
                general_knowledge=general_knowledge,
                math=math,
                total=total,
                top1_course=predictions[0]['course'] if len(predictions) > 0 else None,
                top1_confidence=predictions[0]['confidence_raw'] if len(predictions) > 0 else None,
                top2_course=predictions[1]['course'] if len(predictions) > 1 else None,
                top2_confidence=predictions[1]['confidence_raw'] if len(predictions) > 1 else None,
                top3_course=predictions[2]['course'] if len(predictions) > 2 else None,
                top3_confidence=predictions[2]['confidence_raw'] if len(predictions) > 2 else None,
            )
            db.session.add(pred_record)
            db.session.commit()

        except Exception as exc:
            app.logger.error('Prediction error: %s', exc)
            predictions = [{'course': 'Error – please check your inputs.', 'confidence': 'N/A'}]

    return render_template(
        'index.html',
        predictions=predictions,
        tier_mapping=TIER_MAPPING,
        strand_mapping=STRAND_MAPPING,
    )


# ---------------------------------------------------------------------------
# Dashboard analytics route
# ---------------------------------------------------------------------------

@app.route('/dashboard')
@login_required
def dashboard():
    # All predictions (site-wide counts; personal history for the current user)
    all_predictions = Prediction.query.all()
    user_predictions = (Prediction.query
                        .filter_by(user_id=current_user.id)
                        .order_by(Prediction.created_at.desc())
                        .limit(10)
                        .all())

    total_predictions = len(all_predictions)
    total_users = User.query.count()

    # Course popularity (top-1 prediction only)
    course_counts: dict[str, int] = {}
    for p in all_predictions:
        if p.top1_course:
            course_counts[p.top1_course] = course_counts.get(p.top1_course, 0) + 1
    top_courses = sorted(course_counts.items(), key=lambda x: x[1], reverse=True)[:5]

    # Strand distribution
    strand_counts: dict[str, int] = {}
    for p in all_predictions:
        label = STRAND_MAPPING.get(p.strand, str(p.strand))
        strand_counts[label] = strand_counts.get(label, 0) + 1
    strand_distribution = sorted(strand_counts.items(), key=lambda x: x[1], reverse=True)

    # Average top-1 confidence (overall)
    confidences = [p.top1_confidence for p in all_predictions if p.top1_confidence is not None]
    avg_confidence = (sum(confidences) / len(confidences) * 100) if confidences else 0

    # Recent predictions (last 5 site-wide)
    recent_predictions = (Prediction.query
                          .order_by(Prediction.created_at.desc())
                          .limit(5)
                          .all())

    return render_template(
        'dashboard.html',
        total_predictions=total_predictions,
        total_users=total_users,
        top_courses=top_courses,
        strand_distribution=strand_distribution,
        avg_confidence=f"{avg_confidence:.1f}",
        user_predictions=user_predictions,
        recent_predictions=recent_predictions,
        strand_mapping=STRAND_MAPPING,
    )


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=True)
