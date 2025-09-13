import requests
import os
import time
import sys
import json

# This script requires pandas and openpyxl to create the test Excel file.
# We will check for them and provide a helpful message.
try:
    import pandas as pd
except ImportError:
    print("\nError: 'pandas' library not found. Please run 'pip install pandas' to run this test suite.\n")
    sys.exit(1)

try:
    import openpyxl
except ImportError:
    print("\nError: 'openpyxl' library not found. Please run 'pip install openpyxl' to run this test suite.\n")
    sys.exit(1)


# --- Configuration ---
BASE_URL = "http://127.0.0.1:8000"
UPLOAD_ENDPOINT = f"{BASE_URL}/upload-file/"
CHAT_ENDPOINT = f"{BASE_URL}/chat/"

# --- ANSI escape codes for colored output ---
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{bcolors.HEADER}{bcolors.BOLD}===== {text} ====={bcolors.ENDC}")

def print_pass(text):
    print(f"{bcolors.OKGREEN}[PASS]{bcolors.ENDC} {text}")

def print_fail(text, response_text=None):
    print(f"{bcolors.FAIL}[FAIL]{bcolors.ENDC} {text}")
    if response_text:
        print(f"       {bcolors.WARNING}Response: {response_text[:300]}...{bcolors.ENDC}")


def print_info(text):
    print(f"{bcolors.OKCYAN}[INFO]{bcolors.ENDC} {text}")

def create_test_files():
    """Creates the necessary CSV and XLSX files for testing."""
    print_info("Creating temporary test files...")
    
    # Create employees.csv
    with open("employees.csv", "w") as f:
        f.write("employee_id,employee_name,department\n")
        f.write("101,Anbu,Engineering\n")
        f.write("102,Bala,Sales\n")
        f.write("103,Chandran,Engineering\n")
        f.write("104,Divya,Marketing\n")
    
    # Create a multi-sheet Excel file
    salaries_data = {'employee_id': [101, 102, 103, 104], 'salary_inr': [90000, 75000, 92000, 68000]}
    df_salaries = pd.DataFrame(salaries_data)
    
    locations_data = {'department': ['Engineering', 'Sales', 'Marketing'], 'city': ['Chennai', 'Mumbai', 'Chennai']}
    df_locations = pd.DataFrame(locations_data)

    with pd.ExcelWriter('company_data.xlsx', engine='openpyxl') as writer:
        df_salaries.to_excel(writer, sheet_name='salaries', index=False)
        df_locations.to_excel(writer, sheet_name='locations', index=False)
        
    # Create a dummy text file for failure testing
    with open("invalid_file.txt", "w") as f:
        f.write("This is not a valid file format.")
        
    print_pass("Test files created successfully.")

def cleanup_test_files():
    """Removes the temporary test files."""
    print_info("\nCleaning up temporary test files...")
    for f in ["employees.csv", "company_data.xlsx", "invalid_file.txt"]:
        if os.path.exists(f):
            os.remove(f)
    print_pass("Cleanup complete.")

def run_tests():
    """Executes the entire test suite."""
    
    # --- Test Case 1: Health Check ---
    print_header("Test Case 1: Server Health Check")
    try:
        response = requests.get(BASE_URL, timeout=5)
        if response.status_code == 200:
            print_pass("Server is running and responded with 200 OK.")
        else:
            print_fail(f"Server responded with status code {response.status_code}.")
            return False
    except requests.exceptions.ConnectionError:
        print_fail("Connection to the server failed. Is main.py running?")
        return False
        
    # --- Test Case 2: File Uploads (Rigorous) ---
    print_header("Test Case 2: File Upload Functionality")
    
    # Test 2a: Valid CSV
    try:
        with open('employees.csv', 'rb') as f:
            files = {'file': ('employees.csv', f, 'text/csv')}
            response = requests.post(UPLOAD_ENDPOINT, files=files, timeout=10)
            created_tables_csv = response.json().get("created_tables", [])
            if response.status_code == 200 and "employees" in created_tables_csv:
                print_pass("Successfully uploaded 'employees.csv'.")
            else:
                print_fail("Uploading 'employees.csv' failed.", response.text)
    except Exception as e:
        print_fail(f"An exception occurred during CSV upload test: {e}")

    # Test 2b: Valid multi-sheet Excel
    try:
        with open('company_data.xlsx', 'rb') as f:
            files = {'file': ('company_data.xlsx', f, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')}
            response = requests.post(UPLOAD_ENDPOINT, files=files, timeout=10)
            created_tables = response.json().get("created_tables", [])
            if response.status_code == 200 and 'salaries' in created_tables and 'locations' in created_tables:
                print_pass("Successfully uploaded multi-sheet 'company_data.xlsx' and created both tables.")
            else:
                print_fail("Uploading multi-sheet XLSX failed.", response.text)
    except Exception as e:
        print_fail(f"An exception occurred during XLSX upload test: {e}")
        
    # Test 2c: **HARD TEST** - Invalid file type
    try:
        with open('invalid_file.txt', 'rb') as f:
            files = {'file': ('invalid_file.txt', f, 'text/plain')}
            response = requests.post(UPLOAD_ENDPOINT, files=files, timeout=5)
            if response.status_code == 500 and "Unsupported file type" in response.text:
                print_pass("Correctly rejected an invalid file type (.txt) with a proper error.")
            else:
                print_fail(f"Incorrectly handled invalid file type. Expected 500 with message, got {response.status_code}.", response.text)
    except Exception as e:
        print_fail(f"An exception occurred during invalid file upload test: {e}")

    # --- Test Case 3: Chat Functionality (Rigorous) ---
    print_header("Test Case 3: Chat Functionality")
    all_tables = ["employees", "salaries", "locations"]

    # --- Retry mechanism for chat tests ---
    def run_chat_test(test_name, payload, success_condition, retries=3, delay=2):
        for attempt in range(1, retries + 1):
            try:
                response = requests.post(CHAT_ENDPOINT, json=payload, timeout=120)
                try:
                    response_data = response.json()
                    if response.status_code == 200 and 'answer' in response_data and success_condition(response_data['answer']):
                        print_pass(f"{test_name} answered successfully.")
                        print_info(f"Response: {response_data['answer']}")
                        return
                    else:
                        if attempt < retries:
                            print_info(f"{test_name} failed (attempt {attempt}). Retrying in {delay} seconds...")
                            time.sleep(delay)
                        else:
                            print_fail(f"{test_name} failed after {retries} attempts.", response.text)
                except json.JSONDecodeError:
                    if attempt < retries:
                        print_info(f"{test_name} failed to decode JSON (attempt {attempt}). Retrying in {delay} seconds...")
                        time.sleep(delay)
                    else:
                        print_fail(f"{test_name} failed. Server did not return valid JSON after {retries} attempts.", response.text)
            except Exception as e:
                if attempt < retries:
                    print_info(f"An exception occurred during '{test_name}' test (attempt {attempt}): {e}. Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    print_fail(f"An exception occurred during '{test_name}' test after {retries} attempts: {e}")

    # Test 3a: Simple question
    run_chat_test(
        "Simple question ('how many in engineering')",
        {"table_names": all_tables, "question": "How many employees are in the engineering department?"},
        lambda answer: "2" in answer or "two" in answer.lower()
    )

    # Test 3b: **VERY HARD TEST** - Complex question requiring a JOIN
    run_chat_test(
        "Complex JOIN question (highest salary)",
        {"table_names": all_tables, "question": "What is the name and salary of the employee with the highest salary?"},
        lambda answer: "chandran" in answer.lower() and "92000" in answer
    )

    # Test 3c: **VERY HARD TEST** - Another complex JOIN and aggregation
    run_chat_test(
        "Complex JOIN and aggregation question (city with most departments)",
        {"table_names": all_tables, "question": "Which city has the highest number of departments?"},
        lambda answer: "chennai" in answer.lower()
    )

    return True

if __name__ == "__main__":
    create_test_files()
    try:
        print_info("Waiting 5 seconds for server to potentially restart after uploads...")
        time.sleep(5)
        run_tests()
    finally:
        cleanup_test_files()

