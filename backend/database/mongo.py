from pymongo import MongoClient
from config import MONGO_URI
import sys

# Create client with a short server selection timeout so failures surface promptly
try:
	client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
	# Perform a quick ping to validate authentication/connectivity
	client.admin.command("ping")
except Exception as e:
	# Print a clear message and re-raise so the app fails fast and logs the root cause
	print(f"Failed to connect to MongoDB using MONGO_URI: {e}")
	raise

db = client["cardio_db"]

users_collection = db["users"]
predictions_collection = db["predictions"]
