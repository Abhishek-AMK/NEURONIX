import os

# This path should be the absolute path to your main.py file.
main_py_location = r"C:\Users\AbhishekKulkarni\Desktop\Full-Pipeline\backend\main.py"

# --- DO NOT MODIFY BELOW THIS LINE ---
print("--- STANDALONE PATH DEBUGGER ---")

# 1. Recreate the SCRIPTS_PATH logic from your main.py
try:
    scripts_path = os.path.abspath(os.path.join(os.path.dirname(main_py_location), '..', 'scripts'))
    print(f"Calculated SCRIPTS_PATH: {scripts_path}")
    print(f"Does SCRIPTS_PATH exist?  -> {os.path.isdir(scripts_path)}")
except Exception as e:
    print(f"Error calculating SCRIPTS_PATH: {e}")
    scripts_path = None

if scripts_path:
    # 2. Recreate the path to the specific script
    keyword_script_path = os.path.join(scripts_path, "processing", "keyword_extractor.py")
    print(f"\nConstructed full script path: {keyword_script_path}")

    # 3. Check for existence using different methods
    print(f"Does the 'processing' directory exist? -> {os.path.isdir(os.path.join(scripts_path, 'processing'))}")
    print(f"Does the script file exist (os.path.exists)? -> {os.path.exists(keyword_script_path)}")
    print(f"Is it a file (os.path.isfile)?            -> {os.path.isfile(keyword_script_path)}")

    # 4. List directory contents for debugging
    processing_dir = os.path.join(scripts_path, 'processing')
    if os.path.isdir(processing_dir):
        print(f"\nContents of '{processing_dir}':")
        try:
            for item in os.listdir(processing_dir):
                print(f"- {item}")
        except Exception as e:
            print(f"Could not list directory contents: {e}")
    else:
        print(f"\nCould not list contents because the 'processing' directory does not exist.")

print("\n--- END OF DEBUGGER ---")
