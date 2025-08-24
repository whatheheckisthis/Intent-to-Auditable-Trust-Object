# setup.py (snippet)
from scripts.generate_api_key import generate_key

def security_pipeline():
    print("Initializing security pipeline...")
    key_name = input("Enter default API key filename: ")
    generate_key(key_name)
    print("[✓] Security pipeline complete.")

if __name__ == "__main__":
    security_pipeline()