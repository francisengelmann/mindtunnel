#!/usr/bin/env python3
"""
Text to Speech MP3 Converter using Google Cloud Text-to-Speech API with Gemini-generated paragraph summaries

Prerequisites:
1. Install required packages:
   pip install google-cloud-texttospeech google-ai-generativelanguage

2. Set up Google Cloud credentials:
   - Enable the Text-to-Speech API in your GCP project
   - Enable the Vertex AI API in your GCP project
   - Use the same credentials you use for other GCP services

Usage:
    python text_to_speech.py input.txt output.mp3 --project gcp-superdec
    python text_to_speech.py --help
"""

import argparse
import os
import sys
import re
from pathlib import Path
from typing import Optional, List
from google import genai
from google.cloud import texttospeech


class TextToSpeechConverter:
    def __init__(self, language_code: str = "en-US", voice_name: str = "en-US-Journey-D", 
                 footnote_voice: str = "en-US-Journey-F", narrator_voice: str = "en-US-Studio-O", 
                 project: Optional[str] = None, add_summaries: bool = False, summary_batch_size: int = 4):
        """
        Initialize the Text-to-Speech converter.
        
        Args:
            language_code: Language code (e.g., "en-US", "en-GB")
            voice_name: Main voice name - Journey voice by default (e.g., "en-US-Journey-D")
            footnote_voice: Female voice for title/date/conclusion (e.g., "en-US-Journey-F")
            narrator_voice: Third voice for inline footnotes (e.g., "en-US-Studio-O")
            project: GCP project ID (optional, uses default credentials if not provided)
            add_summaries: Whether to add paragraph summaries
            summary_batch_size: Number of paragraphs to process together for better context (default: 4)
        """
        if project:
            self.client = texttospeech.TextToSpeechClient()
            # The client will automatically use your default credentials
            # Same credentials that work for Vertex AI should work here
        else:
            self.client = texttospeech.TextToSpeechClient()
        
        self.language_code = language_code
        self.voice_name = voice_name
        self.footnote_voice = footnote_voice
        self.narrator_voice = narrator_voice
        self.project = project or "gcp-superdec"  # Default project
        self.add_summaries = add_summaries
        self.summary_batch_size = summary_batch_size
        
        # Initialize Gemini client if summaries are enabled
        self.gemini_client = None
        
        if add_summaries:
            try:
                print(f"Initializing Gemini via Vertex AI for project: {self.project}")
                self.gemini_client = genai.Client(vertexai=True, project=self.project, location='us-central1')
                # Test the model with a simple call
                test_response = self.gemini_client.models.generate_content(
                    model='gemini-2.5-flash-lite',
                    contents='Test'
                )
                print("✓ Gemini model initialized successfully via Vertex AI")
            except Exception as e:
                print(f"Warning: Failed to initialize Gemini via Vertex AI: {str(e)}")
                print("Make sure you have Vertex AI API enabled and proper access to Gemini models")
                print("Continuing without paragraph summaries")
                self.add_summaries = False
        

        
    def _generate_essay_summary(self, full_essay_text: str, essay_title: str) -> Optional[str]:
        """Generate an overall summary of the entire essay using Gemini via Vertex AI."""
        if not self.gemini_client:
            return None
            
        try:
            prompt = f"""You are creating a brief overview summary of Paul Graham's essay "{essay_title}".

FULL ESSAY TEXT:
{full_essay_text[:12000]}  # Use more context for overall summary

Please provide a concise overview of this essay in 2-3 sentences that captures:
1. The main topic or question Paul is addressing
2. His key argument or insight
3. The broader significance or takeaway

This summary will be read aloud at the beginning of the audio version, so make it engaging and accessible. Think of it as setting the stage for what listeners are about to hear.

Overview:"""

            response = self.gemini_client.models.generate_content(
                model='gemini-2.5-flash-lite',
                contents=prompt
            )
            summary = response.text.strip()
            
            # Clean up the summary
            summary = re.sub(r'^(Overview:|The overview is:?|Here\'s the overview:?|Summary:|The summary is:?)', '', summary, flags=re.IGNORECASE).strip()
            
            return summary
            
        except Exception as e:
            print(f"Warning: Failed to generate essay summary: {str(e)}")
            return None

    def _generate_paragraph_summary(self, paragraph: str, essay_title: str, full_essay_context: str = "") -> Optional[str]:
        """Generate a summary for a paragraph using Gemini via Vertex AI."""
        if not self.gemini_client:
            return None
            
        try:
            # Create a more comprehensive prompt with full essay context
            if full_essay_context:
                prompt = f"""You are summarizing a paragraph from Paul Graham's essay "{essay_title}". 

I'm providing you with the full essay context so you can better understand how this specific paragraph fits into the overall argument and themes.

FULL ESSAY CONTEXT:
{full_essay_context[:8000]}  # Limit context to avoid token limits

SPECIFIC PARAGRAPH TO SUMMARIZE:
{paragraph}

Please provide a concise, clear summary of the main idea in this specific paragraph in 1-2 sentences.
Focus on the key insight or argument Paul is making in this paragraph and how it relates to the broader themes of the essay.
Be conversational and accessible, but do not say 'Hi' or something like that at the beginning of a summary.
Also do not explicitley state that is is a summary of that paragraph.

Summary:"""
            else:
                # Fallback to original prompt if no context provided
                prompt = f"""You are summarizing a paragraph from Paul Graham's essay "{essay_title}". 
                
Please provide a concise, clear summary of the main idea in this paragraph in 1-2 sentences. 
Focus on the key insight or argument Paul is making. Be conversational and accessible.

Paragraph:
{paragraph}

Summary:"""

            response = self.gemini_client.models.generate_content(
                model='gemini-2.5-flash-lite',
                contents=prompt
            )
            summary = response.text.strip()
            
            # Clean up the summary
            summary = re.sub(r'^(Summary:|The summary is:?|Here\'s the summary:?)', '', summary, flags=re.IGNORECASE).strip()
            
            return summary
            
        except Exception as e:
            print(f"Warning: Failed to generate summary for paragraph: {str(e)}")
            return None

    def _generate_paragraph_summaries_batch(self, paragraphs: List[str], essay_title: str, full_essay_context: str = "") -> List[Optional[str]]:
        """Generate summaries for multiple paragraphs at once for better context."""
        summaries = []
        
        for i, paragraph in enumerate(paragraphs):
            print(f"  Generating summary for paragraph {i + 1}/{len(paragraphs)}...")
            summary = self._generate_paragraph_summary(paragraph, essay_title, full_essay_context)
            summaries.append(summary)
        
        return summaries
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs, handling various paragraph separators."""
        # Split by double newlines first
        paragraphs = re.split(r'\n\s*\n', text.strip())
        
        # Clean up paragraphs
        cleaned_paragraphs = []
        for para in paragraphs:
            para = para.strip()
            if para and len(para) > 100:  # Only include substantial paragraphs (reduced from 50 to be more inclusive)
                # Remove single line breaks within paragraphs but preserve structure
                para = re.sub(r'\n(?!\s*\n)', ' ', para)
                para = re.sub(r'\s+', ' ', para)  # Normalize whitespace
                cleaned_paragraphs.append(para)
                
        print(f"Split text into {len(cleaned_paragraphs)} substantial paragraphs")
        return cleaned_paragraphs
        
    def convert_text_file(self, input_file: str, output_file: str, 
                         chunk_size: int = 5000) -> bool:
        """
        Convert a text file to MP3 audio with footnote integration and optional paragraph summaries.
        
        Args:
            input_file: Path to input text file
            output_file: Path to output MP3 file
            chunk_size: Maximum characters per API request (Google limit is 5000)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Read the input file
            with open(input_file, 'r', encoding='utf-8') as file:
                text = file.read().strip()
            
            if not text:
                print(f"Error: Input file '{input_file}' is empty")
                return False
            
            print(f"Converting '{input_file}' to '{output_file}'...")
            print(f"Text length: {len(text)} characters")
            if self.add_summaries:
                print("✓ Paragraph summaries will be generated using Gemini")
            else:
                print("✗ Paragraph summaries disabled")
            
            # Process the text to create voice segments
            voice_segments, summary_pairs, title, date = self._create_voice_segments(text)
            output_file = output_file.replace(Path(output_file).stem, title.replace(' ', '_'))
            print(f"Processing {len(voice_segments)} voice segment(s)...")
            
            # Write summary file if summaries were generated
            if self.add_summaries and summary_pairs:
                summary_file = output_file.replace('.mp3', '.txt')
                self._write_summary_file(summary_file, title, date, summary_pairs)
            
            # Convert each segment to audio
            audio_segments = []
            for i, (segment_text, voice) in enumerate(voice_segments, 1):
                print(f"Processing segment {i}/{len(voice_segments)} with voice {voice}...")
                
                # Split long segments into chunks
                if len(segment_text) > chunk_size:
                    chunks = self._split_by_sentences(segment_text, chunk_size)
                else:
                    chunks = [segment_text]
                
                # Process each chunk with the same voice
                for chunk in chunks:
                    audio_content = self._synthesize_speech(chunk, voice)
                    if audio_content:
                        audio_segments.append(audio_content)
                    else:
                        print(f"Error: Failed to synthesize segment {i}")
                        return False
                
                # Add 0.5-second pause between segments (except after the last segment)
                if i < len(voice_segments):
                    pause_audio = self._create_silence(0.5)
                    if pause_audio:
                        audio_segments.append(pause_audio)
            
            # Combine all audio segments
            combined_audio = b''.join(audio_segments)
            
            # Write to output file
            with open(output_file, 'wb') as out_file:
                out_file.write(combined_audio)
            
            print(f"Successfully created '{output_file}'")
            return True
            
        except FileNotFoundError:
            print(f"Error: Input file '{input_file}' not found")
            return False
        except Exception as e:
            print(f"Error: {str(e)}")
            return False
    
    def _create_voice_segments(self, text: str) -> tuple[list[tuple[str, str]], list[tuple[str, str]], str, str]:
        """
        Process text to create voice segments with appropriate voices.
        Returns tuple of (segments, summary_pairs, title, date).
        """
        # Process the text to extract structured content
        segments, summary_pairs, title, date = self._process_essay_text(text)
        
        # Filter out empty segments
        filtered_segments = []
        for segment_text, voice_name in segments:
            if segment_text.strip():
                filtered_segments.append((segment_text.strip(), voice_name))
        
        return filtered_segments, summary_pairs, title, date
    
    def _process_main_content_with_inline_footnotes_and_summaries(self, main_content: str, footnotes: dict, essay_title: str, full_essay_context: str = "") -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        """Process main content, insert footnotes inline, and add paragraph summaries."""
        import re
        
        segments = []
        summary_pairs = []  # Store (paragraph, summary) pairs for output file
        
        # Split content into paragraphs
        paragraphs = self._split_into_paragraphs(main_content)
        print(f"Found {len(paragraphs)} paragraphs to process")
        
        # Process paragraphs in batches for summary generation
        all_summaries = []
        for batch_start in range(0, len(paragraphs), self.summary_batch_size):
            batch_end = min(batch_start + self.summary_batch_size, len(paragraphs))
            batch_paragraphs = paragraphs[batch_start:batch_end]
            
            print(f"Generating summaries for paragraphs {batch_start + 1}-{batch_end} (batch of {len(batch_paragraphs)})...")
            batch_summaries = self._generate_paragraph_summaries_batch(batch_paragraphs, essay_title, full_essay_context)
            all_summaries.extend(batch_summaries)
            
        # Now process each paragraph with its footnotes and summary
        for para_idx, paragraph in enumerate(paragraphs):
            print(f"Processing paragraph {para_idx + 1}/{len(paragraphs)}...")
            
            # Clean paragraph for summary file (remove footnote references for cleaner text)
            clean_paragraph = re.sub(r'\[\d+\]', '', paragraph).strip()
            
            # Process this paragraph for footnotes
            current_text = ""
            parts = re.split(r'(\[\d+\])', paragraph)
            
            for part in parts:
                footnote_match = re.match(r'\[(\d+)\]', part)
                
                if footnote_match:
                    # This is a footnote reference
                    footnote_num = footnote_match.group(1)
                    
                    # Add accumulated main text (if any) with main voice
                    if current_text.strip():
                        segments.append((current_text.strip(), self.voice_name))
                        current_text = ""
                    
                    # Add the footnote with narrator voice (third voice)
                    if footnote_num in footnotes:
                        footnote_text = f"Footnote {footnote_num}: {footnotes[footnote_num]}"
                        segments.append((footnote_text, self.narrator_voice))
                else:
                    # This is regular text, accumulate it
                    current_text += part
            
            # Add any remaining text from this paragraph
            if current_text.strip():
                segments.append((current_text.strip(), self.voice_name))
            
            # Add the pre-generated summary for this paragraph
            if para_idx < len(all_summaries) and all_summaries[para_idx]:
                summary = all_summaries[para_idx]
                segments.append((summary, self.footnote_voice))
                summary_pairs.append((clean_paragraph, summary))
                print(f"Added summary for paragraph {para_idx + 1}: '{summary[:60]}...'")
            else:
                summary_pairs.append((clean_paragraph, ""))
                print(f"No summary available for paragraph {para_idx + 1}")
        
        return segments, summary_pairs
    
    def _process_main_content_with_inline_footnotes(self, main_content: str, footnotes: dict) -> list[tuple[str, str]]:
        """Process main content and insert footnotes inline as they appear (original method)."""
        import re
        
        segments = []
        current_text = ""
        
        # Split text by footnote references while preserving the footnote markers
        parts = re.split(r'(\[\d+\])', main_content)
        
        for part in parts:
            footnote_match = re.match(r'\[(\d+)\]', part)
            
            if footnote_match:
                # This is a footnote reference
                footnote_num = footnote_match.group(1)
                
                # Add accumulated main text (if any) with main voice
                if current_text.strip():
                    segments.append((current_text.strip(), self.voice_name))
                    current_text = ""
                
                # Add the footnote with narrator voice (third voice)
                if footnote_num in footnotes:
                    footnote_text = f"Footnote {footnote_num}: {footnotes[footnote_num]}"
                    segments.append((footnote_text, self.narrator_voice))
            else:
                # This is regular text, accumulate it
                current_text += part
        
        # Add any remaining main content
        if current_text.strip():
            segments.append((current_text.strip(), self.voice_name))
        
        return segments
    
    def _process_essay_text(self, text: str) -> tuple[list[tuple[str, str]], list[tuple[str, str]], str, str]:
        """Process essay text to integrate footnotes and return structured segments plus summary data."""
        import re
        
        lines = text.split('\n')
        
        # Extract title (first non-empty line, clean special characters)
        title = ""
        date = ""
        content_start_index = 0
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            if not title:
                # Clean special characters from title (keep only letters, numbers, spaces, and basic punctuation)
                title = re.sub(r'[^\w\s\-:,.\'"!?()]', '', line).strip()
                content_start_index = i + 1
                continue
            
            # Look for date pattern (e.g., "February 2022")
            date_match = re.search(r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}', line)
            if date_match:
                date = date_match.group(0)
                content_start_index = i + 1
                break
            else:
                # If second line is not a date, it's part of content
                content_start_index = i
                break
        
        # Extract footnotes from the "Notes" section
        footnotes = {}
        notes_started = False
        current_footnote = ""
        current_number = None
        main_content_end_index = len(lines)
        
        for i, line in enumerate(lines):
            if line.strip() == "Notes":
                notes_started = True
                main_content_end_index = i
                continue
            
            if notes_started and line.strip():
                # Look for footnote pattern [1], [2], etc.
                footnote_match = re.match(r'\[(\d+)\]\s*(.*)', line.strip())
                if footnote_match:
                    # Save previous footnote if exists
                    if current_number and current_footnote:
                        footnotes[current_number] = current_footnote.strip()
                    
                    # Start new footnote
                    current_number = footnote_match.group(1)
                    current_footnote = footnote_match.group(2)
                else:
                    # Continue current footnote
                    if current_number:
                        current_footnote += " " + line.strip()
            elif notes_started and not line.strip():
                # End of current footnote
                if current_number and current_footnote:
                    footnotes[current_number] = current_footnote.strip()
                    current_number = None
                    current_footnote = ""
        
        # Save last footnote
        if current_number and current_footnote:
            footnotes[current_number] = current_footnote.strip()
        
        # Get main content (exclude Notes section and acknowledgments)
        main_content_lines = lines[content_start_index:main_content_end_index]
        
        # Remove acknowledgments (lines starting with "Thanks to")
        filtered_content_lines = []
        for line in main_content_lines:
            if not line.strip().startswith("Thanks to"):
                filtered_content_lines.append(line)
            else:
                break  # Stop at acknowledgments
        
        main_content = '\n'.join(filtered_content_lines).strip()
        
        # Build the segments list with (text, voice) tuples
        segments = []
        summary_pairs = []  # For summary output file
        
        # 1. Female voice reads title with "by Paul Graham"
        if title:
            title_with_author = f"{title} by Paul Graham"
            segments.append((title_with_author, self.footnote_voice))
        
        # 2. Female voice reads date
        if date:
            segments.append((date, self.footnote_voice))
        
        # 3. Generate and add essay overview summary if summaries are enabled
        if self.add_summaries:
            print("Generating overall essay summary...")
            essay_summary = self._generate_essay_summary(text, title)
            if essay_summary:
                segments.append((essay_summary, self.footnote_voice))
                print(f"Added essay overview: '{essay_summary[:60]}...'")
            else:
                print("Could not generate essay overview")
        
        # 4. Process main content and integrate footnotes inline (and summaries if enabled)
        if self.add_summaries:
            print(f"Processing main content with summaries enabled")
            # Pass the full original text as context for better summaries
            full_essay_context = text
            content_segments, content_summaries = self._process_main_content_with_inline_footnotes_and_summaries(main_content, footnotes, title, full_essay_context)
            segments.extend(content_segments)
            summary_pairs.extend(content_summaries)
        else:
            print(f"Processing main content without summaries")
            segments.extend(self._process_main_content_with_inline_footnotes(main_content, footnotes))
        
        # 5. Female voice concludes with attribution
        if title:
            conclusion = f"That was {title} by Paul Graham"
            segments.append((conclusion, self.footnote_voice))
        
        return segments, summary_pairs, title, date
    
    def _write_summary_file(self, summary_file: str, title: str, date: str, summary_pairs: list[tuple[str, str]]) -> None:
        """Write the original paragraphs and their summaries to a text file."""
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                # Write header
                f.write(f"{title}\n")
                if date:
                    f.write(f"{date}\n")
                f.write("=" * 80 + "\n")
                f.write("PARAGRAPHS WITH SUMMARIES\n")
                f.write("=" * 80 + "\n\n")
                
                # Write each paragraph with its summary
                for i, (paragraph, summary) in enumerate(summary_pairs, 1):
                    f.write(f"PARAGRAPH {i}:\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"{paragraph}\n\n")
                    
                    if summary:
                        f.write("SUMMARY:\n")
                        f.write(f"{summary}\n")
                    else:
                        f.write("SUMMARY: [No summary generated]\n")
                    
                    f.write("\n" + "=" * 80 + "\n\n")
                
            print(f"Summary file written to: {summary_file}")
            
        except Exception as e:
            print(f"Warning: Could not write summary file '{summary_file}': {str(e)}")

    def _split_by_sentences(self, text: str, chunk_size: int) -> list[str]:
        """Split text by sentences when it's too long."""
        sentences = text.replace('!', '.').replace('?', '.').split('.')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence += '.'
            
            if len(current_chunk) + len(sentence) > chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    # Single sentence too long, add it anyway
                    chunks.append(sentence)
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _create_silence(self, duration_seconds: float) -> Optional[bytes]:
        """Create a silent audio segment of specified duration."""
        try:
            # Create silence by synthesizing periods - shorter for half-second pauses
            silence_text = ". ."  # Two periods for roughly 0.5 seconds
            
            synthesis_input = texttospeech.SynthesisInput(text=silence_text)
            
            # Use the main voice for silence generation
            if self.voice_name.startswith('en-GB'):
                language_code = 'en-GB'
            elif self.voice_name.startswith('en-US'):
                language_code = 'en-US'
            else:
                language_code = self.language_code
            
            voice = texttospeech.VoiceSelectionParams(
                language_code=language_code,
                name=self.voice_name
            )
            
            # Create audio config with very low volume
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3,
                speaking_rate=0.25,  # Minimum allowed rate
                pitch=0.0,
                volume_gain_db=-96.0  # Very quiet (essentially silent)
            )
            
            # Generate the silent audio
            response = self.client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )
            
            return response.audio_content
            
        except Exception as e:
            print(f"Warning: Could not create silence: {str(e)}")
            return None

    def _synthesize_speech(self, text: str, voice_name: Optional[str] = None) -> Optional[bytes]:
        """Synthesize speech for a single text chunk with specified voice."""
        try:
            # Use provided voice or default
            actual_voice = voice_name or self.voice_name
            
            # Determine the correct language code based on the voice
            if actual_voice.startswith('en-GB'):
                language_code = 'en-GB'
            elif actual_voice.startswith('en-US'):
                language_code = 'en-US'
            else:
                # Default to the instance language code
                language_code = self.language_code
            
            # Set up the text input
            synthesis_input = texttospeech.SynthesisInput(text=text)
            
            # Build the voice request
            voice = texttospeech.VoiceSelectionParams(
                language_code=language_code,
                name=actual_voice
            )
            
            # Select the type of audio file
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3,
                speaking_rate=1.0,  # Normal speed
                pitch=0.0,         # Normal pitch
                volume_gain_db=0.0  # Normal volume
            )
            
            # Perform the text-to-speech request
            response = self.client.synthesize_speech(
                input=synthesis_input, 
                voice=voice, 
                audio_config=audio_config
            )
            
            return response.audio_content
            
        except Exception as e:
            print(f"Error synthesizing speech: {str(e)}")
            return None


def main():
    parser = argparse.ArgumentParser(
        description="Convert text files to MP3 using Google Cloud Text-to-Speech with optional Vertex AI Gemini-generated paragraph summaries"
    )
    parser.add_argument("input_file", help="Input text file path")
    parser.add_argument("output_file", help="Output MP3 file path")
    parser.add_argument("--project", default="gcp-superdec", help="GCP Project ID (default: gcp-superdec)")
    parser.add_argument("--language", default="en-US", help="Language code (default: en-US)")
    parser.add_argument("--voice", default="en-US-Journey-D", help="Main voice name - Journey voice by default (default: en-US-Journey-D)")
    parser.add_argument("--footnote-voice", default="en-US-Journey-F", help="Female voice for title, date and conclusion (default: en-US-Journey-F)")
    parser.add_argument("--narrator-voice", default="en-US-Studio-O", help="Third voice for inline footnotes (default: en-US-Studio-O)")
    parser.add_argument("--chunk-size", type=int, default=4500, help="Maximum characters per API request (default: 4500)")
    parser.add_argument("--skip-existing", action="store_true", default=False, help="Skip conversion if MP3 file already exists")
    parser.add_argument("--add-summaries", action="store_true", default=True, help="Add Vertex AI Gemini-generated paragraph summaries")
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist")
        sys.exit(1)
    
    print(f"Using GCP project: {args.project}")
    if args.add_summaries:
        print("Paragraph summaries will be generated using Vertex AI Gemini")
    
    # Check if output file already exists and skip-existing is enabled
    if args.skip_existing and os.path.exists(args.output_file):
        print(f"Output file '{args.output_file}' already exists. Skipping conversion.")
        sys.exit(0)
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert the file
    converter = TextToSpeechConverter(
        language_code=args.language,
        voice_name=args.voice,
        footnote_voice=args.footnote_voice,
        narrator_voice=args.narrator_voice,
        project=args.project,
        add_summaries=args.add_summaries,
        summary_batch_size=4
    )

    success = converter.convert_text_file(
        args.input_file, 
        args.output_file,
        args.chunk_size
    )

    if success:
        file_size = os.path.getsize(args.output_file)
        print(f"Output file size: {file_size:,} bytes")
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
