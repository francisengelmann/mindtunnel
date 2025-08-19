#!/usr/bin/env python3
import os
from pathlib import Path
from text_to_speech import TextToSpeechConverter  # assuming your script is named text_to_speech.py

# Configuration
ESSAYS_DIR = Path("essays")
OUTPUT_DIR = Path("audio")
OUTPUT_DIR.mkdir(exist_ok=True)

# Initialize converter
converter = TextToSpeechConverter(
    language_code="en-US",
    voice_name="en-US-Journey-D",
    footnote_voice="en-US-Journey-F",
    narrator_voice="en-US-Studio-O",
    project="gcp-superdec",
    add_summaries=True,
    summary_batch_size=4
)

# Process all text files
for txt_file in ESSAYS_DIR.glob("*.txt"):
    output_file = OUTPUT_DIR / (txt_file.stem + ".mp3")
    
    if output_file.exists():
        print(f"Skipping '{txt_file}' because '{output_file}' already exists.")
        continue
    
    print(f"\nConverting '{txt_file}' to '{output_file}'...")
    success = converter.convert_text_file(str(txt_file), str(output_file))
    if success:
        print(f"✓ Successfully created '{output_file}'")
    else:
        print(f"✗ Failed to create '{output_file}'")