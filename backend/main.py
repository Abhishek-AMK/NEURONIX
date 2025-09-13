import os
import shutil
import sys
import threading
import uuid
import time
import subprocess
import logging
from flask import Flask, request, jsonify
from logging.handlers import RotatingFileHandler
from flask_cors import CORS

# ==============================================================================
# FLASK APP & LOGGING SETUP
# ==============================================================================
app = Flask(__name__)
CORS(app)

LOG_PATH = os.path.join(os.path.dirname(__file__), "pipeline.log")
if os.path.exists(LOG_PATH):
    os.remove(LOG_PATH)

handler = RotatingFileHandler(LOG_PATH, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(threadName)s | %(message)s')
handler.setFormatter(formatter)
logging.basicConfig(handlers=[handler, logging.StreamHandler(sys.stdout)], level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Pipeline backend starting up.")

# ==============================================================================
# GLOBAL STATE & PATHS
# ==============================================================================
SCRIPTS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
URLS_FILE_PATH = os.path.join(SCRIPTS_PATH, 'scraping', 'urls.txt')

OUTPUT_FOLDERS = [
    os.path.join(SCRIPTS_PATH, 'scraping', 'Justlit-dir'),
    os.path.join(SCRIPTS_PATH, 'ingest', 'output2'),
    os.path.join(SCRIPTS_PATH, 'rag2', 'chunked'),
    os.path.join(SCRIPTS_PATH, 'rag2', 'vector_db'),
    os.path.join(SCRIPTS_PATH, 'rag2', 'query_results'),
    os.path.join(SCRIPTS_PATH, 'processing', 'keywords_output'),
    os.path.join(SCRIPTS_PATH, 'machine_learning', 'ml_output'),
    os.path.join(SCRIPTS_PATH, 'outlier_detection', 'outlier_output')  # Fixed path for outlier detection
]

ACTIVE_PIPELINES = {}

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_script_path(module_type):
    """Maps a module type to its corresponding script path."""
    mapping = {
        "scrape": os.path.join(SCRIPTS_PATH, "scraping", "scraping2.py"),
        "parse_pdf": os.path.join(SCRIPTS_PATH, "ingest", "parsers", "parse_pdf.py"),
        "parse_text": os.path.join(SCRIPTS_PATH, "ingest", "parsers", "parse_text.py"),
        "parse_csv": os.path.join(SCRIPTS_PATH, "ingest", "parsers", "parse_csv.py"),
        "parse_json": os.path.join(SCRIPTS_PATH, "ingest", "parsers", "parse_json.py"),
        "parse_excel": os.path.join(SCRIPTS_PATH, "ingest", "parsers", "parse_excel.py"),
        "parse_docx": os.path.join(SCRIPTS_PATH, "ingest", "parsers", "parse_docx.py"),
        "chunk_texts": os.path.join(SCRIPTS_PATH, "rag2", "chunking.py"),
        "embed_store": os.path.join(SCRIPTS_PATH, "rag2", "embedding.py"),
        "keyword_extraction": os.path.join(SCRIPTS_PATH, "processing", "keyword_extractor.py"),
        "rag_query": os.path.join(SCRIPTS_PATH, "rag2", "rag_query.py"),
        "ml_suite": os.path.join(SCRIPTS_PATH, "machine_learning", "ml.py"),
        "outlier_detection": os.path.join(SCRIPTS_PATH, "outlier_detection", "outlier_detection.py"),  # Fixed path
    }
    return mapping.get(module_type)

def cleanup_pre_run():
    """Cleans up all output folders and the urls.txt file before a pipeline run."""
    logger.info("Cleaning up output folders and URL file before pipeline run.")
    for folder in OUTPUT_FOLDERS:
        try:
            if os.path.exists(folder):
                shutil.rmtree(folder)
            os.makedirs(folder, exist_ok=True)
        except Exception as e:
            logger.error(f"Error cleaning/creating folder {folder}: {e}")
    
    try:
        if os.path.exists(URLS_FILE_PATH):
            os.remove(URLS_FILE_PATH)
            logger.info(f"Removed old urls.txt file.")
    except Exception as e:
        logger.error(f"Error removing urls.txt file: {e}")

def run_script(script_path, params):
    """Executes a given script in a subprocess."""
    env = os.environ.copy()
    for key, value in params.items():
        env[str(key)] = str(value)
    
    logger.info(f"Running script: {os.path.basename(script_path)} with params: {params}")
    
    try:
        start_time = time.time()
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True, text=True, env=env, timeout=10000, check=False
        )
        elapsed = time.time() - start_time
        logger.info(f"Script {os.path.basename(script_path)} finished in {elapsed:.2f}s with code {result.returncode}.")
        
        if result.returncode == 0:
            return True, result.stdout, ""
        else:
            return False, result.stdout, result.stderr
    except Exception as e:
        logger.error(f"Exception running script {script_path}: {e}", exc_info=True)
        return False, "", str(e)

def build_execution_order(modules):
    """Sorts modules by a predefined strict pipeline order with proper error handling."""
    # DEBUG: Log all modules to identify problematic ones
    logger.info(f"DEBUG: Received {len(modules)} modules for execution order")
    for i, module in enumerate(modules):
        logger.info(f"DEBUG: Module {i}: {module}")
        if not isinstance(module, dict):
            logger.error(f"DEBUG: Module {i} is not a dictionary: {module}")
        elif 'type' not in module:
            logger.error(f"DEBUG: Module {i} is missing 'type' field: {module}")
        elif 'id' not in module:
            logger.error(f"DEBUG: Module {i} is missing 'id' field: {module}")
    
    type_order = [
        "scrape",
        "parse_pdf", "parse_csv", "parse_excel", "parse_json", "parse_docx", "parse_text",
        "outlier_detection",  # Added outlier detection after parsing modules
        "chunk_texts",
        "embed_store",
        "keyword_extraction",
        "rag_query",
        "ml_suite"
    ]
    
    # Safe version with error handling
    ordered_ids = []
    for t in type_order:
        for m in modules:
            if not isinstance(m, dict):
                logger.error(f"Invalid module data (not dict): {m}")
                continue
            if 'type' not in m:
                logger.error(f"Module missing 'type' field: {m}")
                continue
            if 'id' not in m:
                logger.error(f"Module missing 'id' field: {m}")
                continue
            if m['type'] == t:
                ordered_ids.append(m['id'])
    
    # Add any remaining modules not in type_order
    for m in modules:
        if isinstance(m, dict) and 'id' in m and m['id'] not in ordered_ids:
            ordered_ids.append(m['id'])
    
    logger.info(f"Execution order determined: {ordered_ids}")
    return ordered_ids

# ==============================================================================
# PIPELINE WORKER THREAD
# ==============================================================================

def pipeline_worker(pipeline_id, modules, connections):
    logger.info(f"PIPELINE_WORKER ({pipeline_id}): Thread started.")
    pipeline_status = ACTIVE_PIPELINES[pipeline_id]
    pipeline_status["status"] = "running"
    
    # Validate modules before processing
    for i, module in enumerate(modules):
        if not isinstance(module, dict):
            error_msg = f"Module {i} is not a dictionary: {module}"
            logger.error(error_msg)
            pipeline_status["status"] = "failed"
            pipeline_status["error"] = error_msg
            ACTIVE_PIPELINES[pipeline_id] = pipeline_status
            return
        if 'type' not in module:
            error_msg = f"Module {i} missing 'type' field: {module}"
            logger.error(error_msg)
            pipeline_status["status"] = "failed"
            pipeline_status["error"] = error_msg
            ACTIVE_PIPELINES[pipeline_id] = pipeline_status
            return
        if 'id' not in module:
            error_msg = f"Module {i} missing 'id' field: {module}"
            logger.error(error_msg)
            pipeline_status["status"] = "failed"
            pipeline_status["error"] = error_msg
            ACTIVE_PIPELINES[pipeline_id] = pipeline_status
            return
    
    try:
        cleanup_pre_run()
        execution_order = build_execution_order(modules)
        id_to_module = {m['id']: m for m in modules}
        
        for module_id in execution_order:
            module = id_to_module[module_id]
            mod_type = module['type']
            
            # --- INPUT HANDLING LOGIC ---
            # Wait for URL input if a scrape module exists and urls.txt is missing
            if mod_type == 'scrape' and not os.path.exists(URLS_FILE_PATH):
                logger.info(f"PIPELINE_WORKER ({pipeline_id}): Scrape module needs URL input.")
                pipeline_status.update({
                    "status": "waiting_for_input",
                    "waiting_module": module_id,
                    "modules": {**pipeline_status.get("modules", {}), module_id: {"status": "waiting_for_input"}}
                })
                ACTIVE_PIPELINES[pipeline_id] = pipeline_status
                
                while ACTIVE_PIPELINES.get(pipeline_id, {}).get("status") == "waiting_for_input":
                    time.sleep(0.5)
            
            # Wait for Question input if a RAG module exists with no question
            elif mod_type == 'rag_query' and not module.get("parameters", {}).get('question', '').strip():
                logger.info(f"PIPELINE_WORKER ({pipeline_id}): RAG module needs question input.")
                pipeline_status.update({
                    "status": "waiting_for_input",
                    "waiting_module": module_id,
                    "modules": {**pipeline_status.get("modules", {}), module_id: {"status": "waiting_for_input"}}
                })
                ACTIVE_PIPELINES[pipeline_id] = pipeline_status
                
                while ACTIVE_PIPELINES.get(pipeline_id, {}).get("status") == "waiting_for_input":
                    time.sleep(0.5)
            
            # Wait for file input if a keyword extraction module exists with no file_path
            elif mod_type == 'keyword_extraction' and not module.get("parameters", {}).get('file_path', '').strip():
                logger.info(f"PIPELINE_WORKER ({pipeline_id}): Keyword extraction module needs file input.")
                pipeline_status.update({
                    "status": "waiting_for_input",
                    "waiting_module": module_id,
                    "modules": {**pipeline_status.get("modules", {}), module_id: {"status": "waiting_for_input"}}
                })
                ACTIVE_PIPELINES[pipeline_id] = pipeline_status
                
                while ACTIVE_PIPELINES.get(pipeline_id, {}).get("status") == "waiting_for_input":
                    time.sleep(0.5)
            
            # Wait for file input and config if ML Suite module exists with no file_path
            elif mod_type == 'ml_suite' and not module.get("parameters", {}).get('file_path', '').strip():
                logger.info(f"PIPELINE_WORKER ({pipeline_id}): ML Suite module needs file input and configuration.")
                pipeline_status.update({
                    "status": "waiting_for_input",
                    "waiting_module": module_id,
                    "modules": {**pipeline_status.get("modules", {}), module_id: {"status": "waiting_for_input"}}
                })
                ACTIVE_PIPELINES[pipeline_id] = pipeline_status
                
                while ACTIVE_PIPELINES.get(pipeline_id, {}).get("status") == "waiting_for_input":
                    time.sleep(0.5)
            
            # NEW: Wait for CSV file input if outlier detection module exists with no file_path
            elif mod_type == 'outlier_detection' and not module.get("parameters", {}).get('file_path', '').strip():
                logger.info(f"PIPELINE_WORKER ({pipeline_id}): Outlier detection module needs CSV file input.")
                pipeline_status.update({
                    "status": "waiting_for_input",
                    "waiting_module": module_id,
                    "modules": {**pipeline_status.get("modules", {}), module_id: {"status": "waiting_for_input"}}
                })
                ACTIVE_PIPELINES[pipeline_id] = pipeline_status
                
                while ACTIVE_PIPELINES.get(pipeline_id, {}).get("status") == "waiting_for_input":
                    time.sleep(0.5)
            
            # Update parameters from user input
            updated_params = ACTIVE_PIPELINES[pipeline_id].get("updated_params", {})
            if "question" in updated_params:
                module["parameters"]["question"] = updated_params["question"]
            if "file_path" in updated_params:
                module["parameters"]["file_path"] = updated_params["file_path"]
            
            # Update ML-specific parameters
            if mod_type == 'ml_suite':
                ml_params = ["task_type", "target_column", "module", "model_type", "unsup_module",
                           "unsup_model", "test_size", "scale", "n_clusters", "n_components", "contamination", "n_jobs"]
                for param in ml_params:
                    if param in updated_params:
                        module["parameters"][param] = updated_params[param]
            
            # NEW: Update outlier detection specific parameters
            if mod_type == 'outlier_detection':
                outlier_params = ["columns"]
                for param in outlier_params:
                    if param in updated_params:
                        module["parameters"][param] = updated_params[param]
            
            # --- SCRIPT EXECUTION ---
            script_path = get_script_path(mod_type)
            logger.info(f"Looking for script at: {script_path}")
            
            if not script_path or not os.path.isfile(script_path):
                raise FileNotFoundError(f"Script not found for module type '{mod_type}' at {script_path}")
            
            params = module.get("parameters", {})
            pipeline_status["modules"][module_id] = {"status": "running", "message": "Executing..."}
            ACTIVE_PIPELINES[pipeline_id] = pipeline_status
            
            success, out, err = run_script(script_path, params)
            
            if not success:
                raise ChildProcessError(f"Module '{mod_type}' failed. Error: {err}")
            
            pipeline_status["modules"][module_id] = {"status": "completed", "message": out.strip()[-2000:]}
            pipeline_status["results"][module_id] = {"output": out.strip()}
            ACTIVE_PIPELINES[pipeline_id] = pipeline_status
        
        pipeline_status["status"] = "completed"
        logger.info(f"PIPELINE_WORKER ({pipeline_id}): Pipeline completed successfully.")
        
    except Exception as e:
        logger.error(f"PIPELINE_WORKER ({pipeline_id}): Pipeline failed. Error: {e}", exc_info=True)
        pipeline_status["status"] = "failed"
        pipeline_status["error"] = str(e)
        ACTIVE_PIPELINES[pipeline_id] = pipeline_status

# ==============================================================================
# API ENDPOINTS
# ==============================================================================

@app.route("/api/pipeline/execute", methods=["POST"])
def api_pipeline_execute():
    data = request.json
    pipeline_id = str(uuid.uuid4())
    logger.info(f"API: Received pipeline execution request. ID: {pipeline_id}")
    
    ACTIVE_PIPELINES[pipeline_id] = {"status": "pending", "modules": {}, "results": {}}
    
    thread = threading.Thread(
        target=pipeline_worker,
        args=(pipeline_id, data.get("modules", []), data.get("connections", [])),
        name=f"Pipeline-{pipeline_id[:8]}",
        daemon=True
    )
    thread.start()
    
    return jsonify({"pipeline_id": pipeline_id, "status": "pending"})

@app.route("/api/pipeline/status/<pipeline_id>", methods=["GET"])
def api_pipeline_status(pipeline_id):
    status = ACTIVE_PIPELINES.get(pipeline_id)
    if status is None:
        return jsonify({"error": "Pipeline not found"}), 404
    return jsonify(status)

@app.route("/api/pipeline/results/<pipeline_id>", methods=["GET"])
def api_pipeline_results(pipeline_id):
    status = ACTIVE_PIPELINES.get(pipeline_id)
    if status is None:
        return jsonify({"error": "Pipeline not found"}), 404
    
    logger.info(f"API: Final results requested for pipeline {pipeline_id}.")
    return jsonify(status)

@app.route("/api/pipeline/update_urls/<pipeline_id>", methods=["POST"])
def api_update_urls(pipeline_id):
    pipeline_status = ACTIVE_PIPELINES.get(pipeline_id)
    if not pipeline_status:
        return jsonify({"error": "Pipeline not found"}), 404
    
    if pipeline_status.get("status") != "waiting_for_input":
        return jsonify({"error": "Pipeline not waiting"}), 400
    
    data = request.json
    urls_data = data.get("urls", [])
    
    try:
        os.makedirs(os.path.dirname(URLS_FILE_PATH), exist_ok=True)
        with open(URLS_FILE_PATH, 'w', encoding='utf-8') as f:
            for entry in urls_data:
                f.write(f"{entry.get('url', '').strip()},{entry.get('depth', 0)}\n")
        
        logger.info(f"API: URLs updated for pipeline {pipeline_id}. Waking up worker.")
        pipeline_status["status"] = "running"
        pipeline_status.pop("waiting_module", None)
        ACTIVE_PIPELINES[pipeline_id] = pipeline_status
        
        return jsonify({"status": "success"})
    except Exception as e:
        logger.error(f"API: Error writing URLs for pipeline {pipeline_id}: {e}")
        return jsonify({"error": f"Failed to write URLs: {str(e)}"}), 500

@app.route("/api/pipeline/update_question/<pipeline_id>", methods=["POST"])
def api_update_question(pipeline_id):
    pipeline_status = ACTIVE_PIPELINES.get(pipeline_id)
    if not pipeline_status:
        return jsonify({"error": "Pipeline not found"}), 404
    
    if pipeline_status.get("status") != "waiting_for_input":
        return jsonify({"error": "Pipeline not waiting"}), 400
    
    data = request.json
    question = data.get("question", "")
    
    logger.info(f"API: Question updated for pipeline {pipeline_id}. Waking up worker.")
    pipeline_status["updated_params"] = {"question": question}
    pipeline_status["status"] = "running"
    pipeline_status.pop("waiting_module", None)
    ACTIVE_PIPELINES[pipeline_id] = pipeline_status
    
    return jsonify({"status": "success"})

# Existing keyword extraction endpoints
@app.route("/api/pipeline/upload_keyword_file/<pipeline_id>", methods=["POST"])
def api_upload_keyword_file(pipeline_id):
    try:
        pipeline_status = ACTIVE_PIPELINES.get(pipeline_id)
        if not pipeline_status:
            return jsonify({"error": "Pipeline not found"}), 404
        
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        uploaded_file = request.files['file']
        if uploaded_file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Create upload directory
        upload_dir = os.path.join(SCRIPTS_PATH, "processing", "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        
        # Save file
        file_path = os.path.join(upload_dir, uploaded_file.filename)
        uploaded_file.save(file_path)
        
        logger.info(f"File uploaded successfully: {file_path}")
        return jsonify({"file_path": file_path, "filename": uploaded_file.filename})
        
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route("/api/pipeline/update_keyword_file/<pipeline_id>", methods=["POST"])
def api_update_keyword_file(pipeline_id):
    pipeline_status = ACTIVE_PIPELINES.get(pipeline_id)
    if not pipeline_status:
        return jsonify({"error": "Pipeline not found"}), 404
    
    if pipeline_status.get("status") != "waiting_for_input":
        return jsonify({"error": "Pipeline not waiting"}), 400
    
    data = request.json
    file_path = data.get("file_path", "")
    
    logger.info(f"API: File path updated for pipeline {pipeline_id}. Waking up worker.")
    pipeline_status["updated_params"] = {"file_path": file_path}
    pipeline_status["status"] = "running"
    pipeline_status.pop("waiting_module", None)
    ACTIVE_PIPELINES[pipeline_id] = pipeline_status
    
    return jsonify({"status": "success"})

# Existing ML Suite endpoints
@app.route("/api/pipeline/upload_ml_file/<pipeline_id>", methods=["POST"])
def api_upload_ml_file(pipeline_id):
    try:
        pipeline_status = ACTIVE_PIPELINES.get(pipeline_id)
        if not pipeline_status:
            return jsonify({"error": "Pipeline not found"}), 404
        
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        uploaded_file = request.files['file']
        if uploaded_file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Validate file extension
        allowed_extensions = {'.csv', '.json', '.xlsx'}
        file_ext = os.path.splitext(uploaded_file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            return jsonify({"error": f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"}), 400
        
        # Save to machine_learning/uploads
        upload_dir = os.path.join(SCRIPTS_PATH, "machine_learning", "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        
        # Save file
        file_path = os.path.join(upload_dir, uploaded_file.filename)
        uploaded_file.save(file_path)
        
        logger.info(f"ML file uploaded successfully: {file_path}")
        return jsonify({"file_path": file_path, "filename": uploaded_file.filename})
        
    except Exception as e:
        logger.error(f"Error uploading ML file: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route("/api/pipeline/update_ml_config/<pipeline_id>", methods=["POST"])
def api_update_ml_config(pipeline_id):
    pipeline_status = ACTIVE_PIPELINES.get(pipeline_id)
    if not pipeline_status:
        return jsonify({"error": "Pipeline not found"}), 404
    
    if pipeline_status.get("status") != "waiting_for_input":
        return jsonify({"error": "Pipeline not waiting"}), 400
    
    data = request.json
    
    # Extract ML configuration parameters
    ml_config = {
        "file_path": data.get("file_path", ""),
        "task_type": data.get("task_type", "supervised"),
        "target_column": data.get("target_column", ""),
        "module": data.get("module", "regression"),
        "model_type": data.get("model_type", "linear"),
        "unsup_module": data.get("unsup_module", "clustering"),
        "unsup_model": data.get("unsup_model", "kmeans"),
        "test_size": data.get("test_size", "0.2"),
        "scale": data.get("scale", "true"),
        "n_clusters": data.get("n_clusters", "3"),
        "n_components": data.get("n_components", "2"),
        "contamination": data.get("contamination", "0.05"),
        "n_jobs": data.get("n_jobs", "-1")
    }
    
    logger.info(f"API: ML configuration updated for pipeline {pipeline_id}. Waking up worker.")
    pipeline_status["updated_params"] = ml_config
    pipeline_status["status"] = "running"
    
    # Update the correct module's status
    ml_module_id = pipeline_status.get("waiting_module")
    if ml_module_id and ml_module_id in pipeline_status.get("modules", {}):
        pipeline_status["modules"][ml_module_id] = {"status": "running"}
    
    pipeline_status.pop("waiting_module", None)
    ACTIVE_PIPELINES[pipeline_id] = pipeline_status
    
    return jsonify({"status": "success"})

# NEW: Outlier Detection endpoints
@app.route("/api/pipeline/upload_outlier_file/<pipeline_id>", methods=["POST"])
def api_upload_outlier_file(pipeline_id):
    try:
        pipeline_status = ACTIVE_PIPELINES.get(pipeline_id)
        if not pipeline_status:
            return jsonify({"error": "Pipeline not found"}), 404
        
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        uploaded_file = request.files['file']
        if uploaded_file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Validate file extension - only CSV for outlier detection
        file_ext = os.path.splitext(uploaded_file.filename)[1].lower()
        if file_ext != '.csv':
            return jsonify({"error": "Only CSV files are supported for outlier detection"}), 400
        
        # Create upload directory for outlier detection
        upload_dir = os.path.join(SCRIPTS_PATH, "processing", "outlier_uploads")
        os.makedirs(upload_dir, exist_ok=True)
        
        # Save file
        file_path = os.path.join(upload_dir, uploaded_file.filename)
        uploaded_file.save(file_path)
        
        logger.info(f"Outlier detection CSV file uploaded successfully: {file_path}")
        return jsonify({"file_path": file_path, "filename": uploaded_file.filename})
        
    except Exception as e:
        logger.error(f"Error uploading outlier detection file: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route("/api/pipeline/update_outlier_config/<pipeline_id>", methods=["POST"])
def api_update_outlier_config(pipeline_id):
    pipeline_status = ACTIVE_PIPELINES.get(pipeline_id)
    if not pipeline_status:
        return jsonify({"error": "Pipeline not found"}), 404
    
    if pipeline_status.get("status") != "waiting_for_input":
        return jsonify({"error": "Pipeline not waiting"}), 400
    
    data = request.json
    
    # Extract outlier detection configuration parameters
    outlier_config = {
        "file_path": data.get("file_path", ""),
        "columns": data.get("columns", "")  # Comma-separated column names
    }
    
    logger.info(f"API: Outlier detection configuration updated for pipeline {pipeline_id}. Waking up worker.")
    pipeline_status["updated_params"] = outlier_config
    pipeline_status["status"] = "running"
    
    # Update the correct module's status
    outlier_module_id = pipeline_status.get("waiting_module")
    if outlier_module_id and outlier_module_id in pipeline_status.get("modules", {}):
        pipeline_status["modules"][outlier_module_id] = {"status": "running"}
    
    pipeline_status.pop("waiting_module", None)
    ACTIVE_PIPELINES[pipeline_id] = pipeline_status
    
    return jsonify({"status": "success"})

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    logger.info("Starting Flask app on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
