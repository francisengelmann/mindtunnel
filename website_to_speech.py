#!/usr/bin/env python3
"""
Website to Speech Converter using Google Cloud Text-to-Speech API

This script extracts the main content from a website and converts it to an MP3 audio file.
It removes navigation, headers, footers, and other non-content elements.

Prerequisites:
1. Install required packages:
   pip install google-cloud-texttospeech beautifulsoup4 requests readability-lxml

2. Set up Google Cloud credentials:
   - Enable the Text-to-Speech API in your GCP project
   - Set GOOGLE_APPLICATION_CREDENTIALS environment variable

Usage:
    python website_to_speech.py https://example.com/article output.mp3 --project gcp-superdec
    python website_to_speech.py --help
"""

import argparse
import os
import sys
import re
from pathlib import Path
from typing import Optional
import requests
from bs4 import BeautifulSoup
from readability import Document
from google.cloud import texttospeech


class WebsiteToSpeechConverter:
    def __init__(self, language_code: str = "en-US", voice_name: str = "en-US-Journey-D",
                 narrator_voice: str = "en-US-Journey-F", project: Optional[str] = None):
        """
        Initialize the Website-to-Speech converter.
        
        Args:
            language_code: Language code (e.g., "en-US", "en-GB")
            voice_name: Main voice name for content (e.g., "en-US-Journey-D")
            narrator_voice: Voice for title and conclusion (e.g., "en-US-Journey-F")
            project: GCP project ID (optional, uses default credentials if not provided)
        """
        self.client = texttospeech.TextToSpeechClient()
        self.language_code = language_code
        self.voice_name = voice_name
        self.narrator_voice = narrator_voice
        self.project = project or "gcp-superdec"
        
    def extract_content(self, url: str) -> tuple[str, str, str]:
        """
        Extract clean content from a website URL.
        
        Args:
            url: The website URL to extract content from
            
        Returns:
            Tuple of (title, author, content)
        """
        try:
            print(f"Fetching content from: {url}")
            
            # Fetch the webpage
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Use readability to extract main content
            doc = Document(response.text)
            title = doc.title()
            html_content = doc.summary()
            
            # Parse with BeautifulSoup to clean up
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                element.decompose()
            
            # Extract text
            text = soup.get_text(separator='\n\n')
            
            # Clean up the text
            lines = []
            for line in text.split('\n'):
                line = line.strip()
                if line and len(line) > 3:  # Skip very short lines
                    lines.append(line)
            
            content = '\n\n'.join(lines)
            
            # Try to extract author from meta tags
            author = ""
            full_soup = BeautifulSoup(response.text, 'html.parser')
            author_meta = full_soup.find('meta', attrs={'name': 'author'})
            if author_meta and author_meta.get('content'):
                author = author_meta.get('content')
            else:
                # Try other common author meta tags
                for meta_name in ['article:author', 'twitter:creator', 'DC.creator']:
                    author_meta = full_soup.find('meta', attrs={'property': meta_name})
                    if not author_meta:
                        author_meta = full_soup.find('meta', attrs={'name': meta_name})
                    if author_meta and author_meta.get('content'):
                        author = author_meta.get('content')
                        # Clean Twitter handles
                        author = author.replace('@', '')
                        break
            
            print(f"✓ Extracted content successfully")
            print(f"  Title: {title}")
            print(f"  Author: {author if author else '(not found)'}")
            print(f"  Content length: {len(content)} characters")
            
            return title, author, content
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching URL: {str(e)}")
            raise
        except Exception as e:
            print(f"Error extracting content: {str(e)}")
            raise
    
    def convert_url_to_audio(self, url: str, output_file: str, chunk_size: int = 4500) -> bool:
        """
        Convert a website URL to an MP3 audio file.
        
        Args:
            url: The website URL to convert
            output_file: Path to output MP3 file
            chunk_size: Maximum characters per API request (Google limit is 5000)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Extract content from the website
            title, author, content = self.extract_content(url)
            
            if not content:
                print("Error: No content extracted from the website")
                return False
            
            # Create voice segments
            print(f"Creating audio segments...")
            voice_segments = self._create_voice_segments(title, author, content)
            
            print(f"Processing {len(voice_segments)} voice segment(s)...")
            
            # Convert each segment to audio
            audio_segments = []
            for i, (segment_text, voice) in enumerate(voice_segments, 1):
                print(f"Processing segment {i}/{len(voice_segments)}...")
                
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
                
                # Add 0.5-second pause between segments
                if i < len(voice_segments):
                    pause_audio = self._create_silence(0.5)
                    if pause_audio:
                        audio_segments.append(pause_audio)
            
            # Combine all audio segments
            combined_audio = b''.join(audio_segments)
            
            # Write to output file
            with open(output_file, 'wb') as out_file:
                out_file.write(combined_audio)
            
            file_size = os.path.getsize(output_file)
            print(f"✓ Successfully created '{output_file}'")
            print(f"  File size: {file_size:,} bytes")
            return True
            
        except Exception as e:
            print(f"Error: {str(e)}")
            return False
    
    def _create_voice_segments(self, title: str, author: str, content: str) -> list[tuple[str, str]]:
        """
        Create voice segments with appropriate voices for different parts.
        
        Returns:
            List of (text, voice_name) tuples
        """
        segments = []
        
        # 1. Narrator voice introduces the article
        if title:
            title_text = f"{title}"
            if author:
                title_text += f" by {author}"
            segments.append((title_text, self.narrator_voice))
        
        # 2. Split content into paragraphs and use main voice
        paragraphs = self._split_into_paragraphs(content)
        print(f"Split content into {len(paragraphs)} paragraphs")
        
        for paragraph in paragraphs:
            if paragraph.strip():
                segments.append((paragraph.strip(), self.voice_name))
        
        return segments
    
    def _split_into_paragraphs(self, text: str) -> list[str]:
        """Split text into paragraphs."""
        # Split by double newlines
        paragraphs = re.split(r'\n\s*\n', text.strip())
        
        # Clean up paragraphs
        cleaned_paragraphs = []
        for para in paragraphs:
            para = para.strip()
            if para and len(para) > 50:  # Only include substantial paragraphs
                # Normalize whitespace
                para = re.sub(r'\s+', ' ', para)
                cleaned_paragraphs.append(para)
        
        return cleaned_paragraphs
    
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
            silence_text = ". ."  # Two periods for roughly 0.5 seconds
            
            synthesis_input = texttospeech.SynthesisInput(text=silence_text)
            
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
            
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3,
                speaking_rate=0.25,
                pitch=0.0,
                volume_gain_db=-96.0  # Very quiet (essentially silent)
            )
            
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
            actual_voice = voice_name or self.voice_name
            
            # Determine the correct language code based on the voice
            if actual_voice.startswith('en-GB'):
                language_code = 'en-GB'
            elif actual_voice.startswith('en-US'):
                language_code = 'en-US'
            else:
                language_code = self.language_code
            
            synthesis_input = texttospeech.SynthesisInput(text=text)
            
            voice = texttospeech.VoiceSelectionParams(
                language_code=language_code,
                name=actual_voice
            )
            
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3,
                speaking_rate=1.0,
                pitch=0.0,
                volume_gain_db=0.0
            )
            
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
        description="Convert website content to MP3 using Google Cloud Text-to-Speech"
    )
    parser.add_argument("url", help="Website URL to convert")
    parser.add_argument("output_file", help="Output MP3 file path")
    parser.add_argument("--project", default="gcp-superdec", help="GCP Project ID (default: gcp-superdec)")
    parser.add_argument("--language", default="en-US", help="Language code (default: en-US)")
    parser.add_argument("--voice", default="en-US-Journey-D", help="Main voice for content (default: en-US-Journey-D)")
    parser.add_argument("--narrator-voice", default="en-US-Journey-F", help="Voice for title and conclusion (default: en-US-Journey-F)")
    parser.add_argument("--chunk-size", type=int, default=4500, help="Maximum characters per API request (default: 4500)")
    
    args = parser.parse_args()
    
    # Validate URL
    if not args.url.startswith(('http://', 'https://')):
        print("Error: URL must start with http:// or https://")
        sys.exit(1)
    
    print(f"Using GCP project: {args.project}")
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert the website to audio
    converter = WebsiteToSpeechConverter(
        language_code=args.language,
        voice_name=args.voice,
        narrator_voice=args.narrator_voice,
        project=args.project
    )
    
    success = converter.convert_url_to_audio(
        args.url,
        args.output_file,
        args.chunk_size
    )
    
    if success:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
