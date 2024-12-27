from flask import Flask, request, render_template, send_file, jsonify
import os
from image_processing import process_image, save_image, blocks_save_image
import threading
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
PROCESSED_FOLDER = "processed/image_output"
UNIQUE_BLOCKS_FOLDER = "processed/unique_blocks_output"
palette_data = {} 

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(UNIQUE_BLOCKS_FOLDER, exist_ok=True)

# Dictionary to track the status of unique block processing
processing_status = {}

def process_unique_blocks(input_path, blocks_output_path, file_key, custom_palette):
    """
    Processes unique blocks in the background and updates their status.
    """
    try:
        # Generate the unique blocks and save them
        _, _, unique_blocks = process_image(input_path, custom_palette)
        blocks_save_image(unique_blocks, blocks_output_path, save_as_grid=True)
        # Mark the process as completed
        processing_status[file_key] = "ready"
    except Exception as e:
        print(f"Error processing unique blocks: {e}")
        processing_status[file_key] = "error"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    # Handle file upload
    file = request.files["image"]
    if not file:
        return "No file selected", 400

    # Save uploaded file
    input_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(input_path)

    # Get custom palette from user input
    custom_palette = request.form.getlist("colors")
    custom_palette = [color for color in custom_palette if color]  # Filter empty inputs

    # Process the main image
    output_image, used_palette, _ = process_image(input_path, custom_palette)
    output_path = os.path.join(PROCESSED_FOLDER, "processed_" + file.filename)
    save_image(output_image, output_path)

    # Store the palette for retrieval
    palette_data[file.filename] = used_palette

    # Start processing unique blocks in the background
    blocks_output_path = os.path.join(UNIQUE_BLOCKS_FOLDER, "unique_blocks_" + file.filename)
    file_key = file.filename  # Use the filename as a key for tracking
    processing_status[file_key] = "processing"

    threading.Thread(
        target=process_unique_blocks,
        args=(input_path, blocks_output_path, file_key, custom_palette)
    ).start()

    # Return the processed image immediately
    return send_file(output_path, mimetype='image/png', as_attachment=False)


@app.route("/unique-blocks", methods=["POST"])
def unique_blocks():
    """
    Endpoint to check the status of unique blocks processing and send the file if ready.
    """
    file_name = request.form.get("file_name")
    if not file_name:
        return "File name not provided", 400

    file_key = file_name  # Use filename as the key
    blocks_output_path = os.path.join(UNIQUE_BLOCKS_FOLDER, "unique_blocks_" + file_name)

    status = processing_status.get(file_key, None)
    if status == "ready":
        # File is ready, send it to the client
        return send_file(blocks_output_path, mimetype='image/png', as_attachment=False)
    elif status == "processing":
        # Still processing
        return jsonify({"status": "processing"}), 202
    elif status == "error":
        # Error occurred during processing
        return jsonify({"status": "error"}), 500
    else:
        # Unknown file key
        return jsonify({"status": "not found"}), 404
    
def rgb_to_hex(color):
    """Convert normalized RGB to hex color."""
    return "#{:02x}{:02x}{:02x}".format(
        int(color[0] * 255), 
        int(color[1] * 255), 
        int(color[2] * 255)
    )

@app.route("/palette", methods=["GET"])
def get_palette():
    file_name = request.args.get("file_name")
    if not file_name or file_name not in palette_data:
        return "Palette not found", 404
    
    # Convert normalized RGB palette to hex
    palette = palette_data[file_name]
    if isinstance(palette, np.ndarray):  # Convert NumPy array to list
        palette = palette.tolist()
    
    hex_palette = [rgb_to_hex(color) for color in palette]
    return jsonify({"used_palette": hex_palette})


if __name__ == "__main__":
    app.run(debug=True, port=8000)
