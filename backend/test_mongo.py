"""
Simple script to verify MongoDB connectivity using the project's config.
Run this after creating `backend/.env` or setting `MONGO_URI` in your environment.
"""
from pymongo import MongoClient
from config import MONGO_URI

def main():
    print("Testing MongoDB connection...")
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    try:
        client.admin.command("ping")
        print("MongoDB connection OK")
    except Exception as e:
        print("MongoDB connection FAILED:\n", e)

if __name__ == "__main__":
    main()
