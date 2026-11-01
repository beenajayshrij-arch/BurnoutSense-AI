
"""
CLI tool to add HR user accounts.
Passwords are stored using salted PBKDF2 hashing.
"""

import json
import os
import hashlib
import secrets

USERS_PATH = "data/hr_users.json"
ITERATIONS = 600000


def hash_password(password):
    salt = secrets.token_hex(16)

    hashed = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode(),
        bytes.fromhex(salt),
        ITERATIONS
    ).hex()

    return f"pbkdf2${ITERATIONS}${salt}${hashed}"


def load_users():
    if not os.path.exists(USERS_PATH):
        return {}

    with open(USERS_PATH) as f:
        return json.load(f)


def save_users(users):
    os.makedirs(os.path.dirname(USERS_PATH), exist_ok=True)

    with open(USERS_PATH, "w") as f:
        json.dump(users, f, indent=2)


def add_user():
    users = load_users()

    username = input("New HR username: ").strip()

    if not username:
        print("Username cannot be empty.")
        return

    if username in users:
        print("Username already exists.")
        return

    password = input("New HR password: ").strip()
    confirm = input("Confirm password: ").strip()

    if password != confirm:
        print("Passwords do not match.")
        return

    if len(password) < 8:
        print("Password must contain at least 8 characters.")
        return

    users[username] = hash_password(password)
    save_users(users)

    print("HR user added successfully.")


if __name__ == "__main__":
    add_user()
