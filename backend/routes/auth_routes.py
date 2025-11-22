from flask import Blueprint, request, jsonify
from database.mongo import users_collection
from utils.hashing import hash_password, verify_password
from utils.token import generate_token

auth = Blueprint("auth", __name__)

@auth.post("/signup")
def signup():
    data = request.json
    email = data["email"]

    if users_collection.find_one({"email": email}):
        return jsonify({"msg": "User already exists"}), 400

    role = data.get("role", "Doctor")

    users_collection.insert_one({
        "name": data["name"],
        "email": data["email"],
        "password": hash_password(data["password"]),
        "role": role,
    })

    return jsonify({"msg": "Signup successful"})


@auth.post("/login")
def login():
    data = request.json
    user = users_collection.find_one({"email": data["email"]})

    if not user:
        return jsonify({"msg": "User not found"}), 404

    if not verify_password(data["password"], user["password"]):
        return jsonify({"msg": "Incorrect password"}), 401

    token = generate_token(user["email"], name=user.get("name"), role=user.get("role"))
    return jsonify({
        "token": token,
        "name": user.get("name"),
        "role": user.get("role", "Doctor"),
    })
