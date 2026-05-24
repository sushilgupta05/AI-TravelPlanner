import os
from dotenv import load_dotenv

# Force load the env file
load_dotenv()

# Print the value to the terminal so we can see what it's REALLY reading
db_url = os.getenv("DATABASE_URL")
print(f"DEBUG: The URL Python is reading is: {db_url}")

if db_url is None:
    print("❌ ERROR: DATABASE_URL is None! Your .env file is not being loaded.")
else:
    print("✅ DATABASE_URL is loaded.")