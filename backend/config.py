import os
from dotenv import load_dotenv

# Load environment variables from a local .env file (if present).
load_dotenv()

# Read secrets from environment when available to avoid committing credentials.
# Set `MONGO_URI` in your environment or replace the default placeholder below.
MONGO_URI = os.environ.get(
	"MONGO_URI",
	"mongodb+srv://lokesh:lokesh0910@cluster0.ac8ssri.mongodb.net/?appName=Cluster0",
)

# JWT secret should also come from env in production
JWT_SECRET = os.environ.get("JWT_SECRET", "YOUR_SECRET_KEY")
