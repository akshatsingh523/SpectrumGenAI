# login.py
from pymongo import MongoClient
import bcrypt
import hashlib
import uuid

# MongoDB connection setup
def get_database():
    CONNECTION_STRING = "mongodb+srv://madhogariaraksha27:HdAlSZloseoSMdNZ@llama.siihara.mongodb.net/user_questions?retryWrites=true&w=majority"
    client = MongoClient(CONNECTION_STRING)
    return client['user_questions']

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(username, password):
    db = get_database()
    users = db.users
    if users.find_one({"username": username}):
        return False
    else:
        user_id = str(uuid.uuid4())  
        hashed_password = hash_password(password)
        user_data = {"user_id":user_id, "username": username, "password": hashed_password}
        users.insert_one(user_data)
        # return True
        return True,user_id 

def check_user(username, password):
    db = get_database()
    users = db.users
    user = users.find_one({"username": username, "password": hash_password(password)})
    if user and 'user_id' in user:
        return user
      
        # return user is not None
    return None

# def create_user(username, password):
#     hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
#     user_data = {'username': username, 'password': hashed_password}
#     user_collection.insert_one(user_data)

# def check_user(username, password):
#     user_found = user_collection.find_one({'username': username})
#     if user_found:
#         return bcrypt.checkpw(password.encode('utf-8'), user_found['password'])
#     return False
if __name__ == "__main__":
    # Quick test to see if it connects to the database and can insert a user
    db = get_database()