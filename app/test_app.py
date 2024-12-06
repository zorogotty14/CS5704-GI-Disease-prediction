import pytest
from app import app, db
from models.users import User

@pytest.fixture(scope="function")
def reset_database():
    """
    Fixture to reset the database before each test.
    Drops all tables, recreates them, and ensures a clean slate.
    """
    with app.app_context():
        db.drop_all()  # Drop all tables
        db.create_all()  # Recreate all tables
        yield  # Provide control to the test function
        db.session.remove()  # Clear the session


@pytest.fixture(scope="function")
def test_client(reset_database):
    """
    Fixture to provide a test client.
    Depends on the reset_database fixture to ensure a clean database for each test.
    """
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def create_test_user(email="test@example.com"):
    """
    Helper function to add a test user to the database.
    """
    user = User(name="Test User", email=email)
    user.set_password("testpassword")
    db.session.add(user)
    db.session.commit()


def test_register(test_client):
    response = test_client.post('/register', data={
        'name': 'New User',
        'email': 'newuser@example.com',
        'password': 'newpassword'
    }, follow_redirects=True)
    assert response.status_code == 200
    assert b'Registration successful! Please log in.' in response.data


def test_login(test_client):
    create_test_user()

    response = test_client.post('/login', data={
        'email': 'test@example.com',
        'password': 'testpassword'
    }, follow_redirects=True)
    assert response.status_code == 200
    assert b'Gastrointestinal disease prediction' in response.data


def test_logout(test_client):
    create_test_user()
    test_client.post('/login', data={
        'email': 'test@example.com',
        'password': 'testpassword'
    }, follow_redirects=True)

    response = test_client.get('/logout', follow_redirects=True)
    assert response.status_code == 200
    assert b'You have been logged out.' in response.data


def test_upload_and_predict(test_client):
    create_test_user()
    test_client.post('/login', data={
        'email': 'test@example.com',
        'password': 'testpassword'
    }, follow_redirects=True)

    # Test uploading an image
    with open('./test_image.jpg', 'rb') as img:
        response = test_client.post('/predict', data={
            'file': (img, 'test_image.jpg')
        }, content_type='multipart/form-data', follow_redirects=True)

    assert response.status_code == 200
    assert b'Predicted gastrointestinal condition:' in response.data
