"""
Transcription Module
Handles audio transcription using OpenAI Whisper
"""

import os
import json
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from loguru import logger
from dataclasses import dataclass
import numpy as np

@dataclass
class TranscriptSegment:
    """Single transcript segment"""
    start: float
    end: float
    text: str
    confidence: float
    words: List[Dict]

class TranscriptionEngine:
    """
    Transcribe audio using OpenAI Whisper
    Supports multiple model sizes and languages
    """
    
    MODEL_SIZES = {
        'tiny': {'params': '39M', 'speed': '~32x', 'memory': '~1GB'},
        'base': {'params': '74M', 'speed': '~16x', 'memory': '~1GB'},
        'small': {'params': '244M', 'speed': '~6x', 'memory': '~2GB'},
        'medium': {'params': '769M', 'speed': '~2x', 'memory': '~5GB'},
        'large': {'params': '1550M', 'speed': '~1x', 'memory': '~10GB'},
    }
    
    def __init__(self, model_size: str = "small", use_gpu: bool = True):
        """
        Initialize transcription engine
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            use_gpu: Whether to use GPU acceleration
        """
        self.model_size = model_size
        self.use_gpu = use_gpu
        self.model = None
        self.device = None
        
        self._load_model()
    
    def _load_model(self):
        """Load Whisper model"""
        try:
            import whisper
            import torch
            
            # Determine device
            if self.use_gpu and torch.cuda.is_available():
                self.device = "cuda"
                logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
            else:
                self.device = "cpu"
                logger.info("Using CPU for transcription")
            
            logger.info(f"Loading Whisper {self.model_size} model on {self.device}...")
            self.model = whisper.load_model(self.model_size, device=self.device)
            logger.success(f"Whisper model loaded successfully")
            
        except ImportError as e:
            logger.error(f"Whisper not installed: {str(e)}")
            logger.info("Install with: pip install openai-whisper")
            raise
        except Exception as e:
            logger.error(f"Error loading Whisper model: {str(e)}")
            raise
    
    def transcribe(self, video_path: str, language: Optional[str] = None) -> Dict[str, Any]:
        """
        Transcribe audio from video file
        
        Args:
            video_path: Path to video file
            language: Language code (e.g., 'en', 'es', 'fr'). Auto-detected if None
            
        Returns:
            Dictionary with transcription results
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        if self.model is None:
            raise RuntimeError("Whisper model not loaded")
        
        logger.info(f"Transcribing: {video_path}")
        
        try:
            # Extract audio if needed
            audio_path = self._extract_audio(video_path)
            
            # Transcribe with word-level timestamps
            result = self.model.transcribe(
                audio_path,
                language=language,
                task="transcribe",
                word_timestamps=True,
                fp16=self.device == "cuda"
            )
            
            # Process results
            segments = []
            all_words = []
            
            for segment in result.get('segments', []):
                seg_data = {
                    'start': segment['start'],
                    'end': segment['end'],
                    'text': segment['text'].strip(),
                    'confidence': self._calculate_segment_confidence(segment),
                    'words': []
                }
                
                # Extract words with timestamps
                if 'words' in segment and segment['words']:
                    for word in segment['words']:
                        word_data = {
                            'word': word['word'].strip(),
                            'start': word['start'],
                            'end': word['end'],
                            'probability': word.get('probability', 1.0)
                        }
                        seg_data['words'].append(word_data)
                        all_words.append(word_data)
                
                segments.append(seg_data)
            
            # Calculate overall stats
            total_duration = segments[-1]['end'] if segments else 0
            total_words = len(all_words)
            avg_confidence = np.mean([s['confidence'] for s in segments]) if segments else 0
            
            transcription_result = {
                'text': result.get('text', ''),
                'segments': segments,
                'words': all_words,
                'language': result.get('language', 'unknown'),
                'duration': total_duration,
                'word_count': total_words,
                'avg_confidence': float(avg_confidence),
                'model': self.model_size,
                'device': self.device
            }
            
            logger.success(f"Transcription complete: {total_words} words, {total_duration:.1f}s")
            
            # Clean up temp audio file
            if audio_path != video_path and os.path.exists(audio_path):
                os.remove(audio_path)
            
            return transcription_result
            
        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            raise
    
    def _extract_audio(self, video_path: str) -> str:
        """
        Extract audio from video file
        
        Args:
            video_path: Path to video file
            
        Returns:
            Path to audio file
        """
        # Check if already an audio file
        audio_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg']
        if Path(video_path).suffix.lower() in audio_extensions:
            return video_path
        
        # Extract using ffmpeg
        temp_dir = tempfile.gettempdir()
        audio_path = os.path.join(temp_dir, f"audio_{Path(video_path).stem}.wav")
        
        try:
            cmd = [
                'ffmpeg', '-y', '-i', video_path,
                '-vn', '-acodec', 'pcm_s16le',
                '-ar', '16000', '-ac', '1',
                audio_path
            ]
            
            subprocess.run(cmd, capture_output=True, check=True)
            logger.info(f"Audio extracted to: {audio_path}")
            return audio_path
            
        except subprocess.CalledProcessError as e:
            logger.warning(f"FFmpeg extraction failed: {str(e)}")
            return video_path  # Return original and let Whisper handle it
        except FileNotFoundError:
            logger.warning("FFmpeg not found, using video file directly")
            return video_path
    
    def _calculate_segment_confidence(self, segment: Dict) -> float:
        """Calculate average confidence for a segment"""
        if 'words' in segment and segment['words']:
            probs = [w.get('probability', 1.0) for w in segment['words']]
            return float(np.mean(probs))
        elif 'avg_logprob' in segment:
            # Convert log probability to confidence
            return float(np.exp(segment['avg_logprob']))
        return 1.0
    
    def transcribe_with_diarization(self, video_path: str) -> Dict[str, Any]:
        """
        Transcribe with speaker diarization (if available)
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with transcription and speaker info
        """
        try:
            import whisperx
            
            # Load audio
            audio = whisperx.load_audio(video_path)
            
            # Transcribe
            result = self.model.transcribe(audio)
            
            # Align
            model_a, metadata = whisperx.load_align_model(
                language_code=result["language"],
                device=self.device
            )
            result = whisperx.align(
                result["segments"],
                model_a,
                metadata,
                audio,
                self.device
            )
            
            # Diarize
            diarize_model = whisperx.DiarizationPipeline(
                use_auth_token="YOUR_HF_TOKEN",
                device=self.device
            )
            diarize_segments = diarize_model(audio)
            result = whisperx.assign_word_speakers(diarize_segments, result)
            
            return result
            
        except ImportError:
            logger.warning("whisperx not installed, using standard transcription")
            return self.transcribe(video_path)
    
    def get_segments_by_time(self, transcription: Dict, start: float, end: float) -> List[Dict]:
        """
        Get transcript segments within a time range
        
        Args:
            transcription: Transcription result
            start: Start time in seconds
            end: End time in seconds
            
        Returns:
            List of segments in the time range
        """
        segments = []
        
        for segment in transcription.get('segments', []):
            seg_start = segment['start']
            seg_end = segment['end']
            
            # Check if segment overlaps with range
            if seg_end >= start and seg_start <= end:
                segments.append(segment)
        
        return segments
    
    def get_text_by_time(self, transcription: Dict, start: float, end: float) -> str:
        """
        Get transcribed text within a time range
        
        Args:
            transcription: Transcription result
            start: Start time in seconds
            end: End time in seconds
            
        Returns:
            Combined text in the time range
        """
        segments = self.get_segments_by_time(transcription, start, end)
        return ' '.join([s['text'] for s in segments])
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Simple sentiment analysis of text
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment analysis
        """
        # Simple keyword-based sentiment
        positive_words = [
            'amazing', 'awesome', 'great', 'excellent', 'fantastic', 'wonderful',
            'love', 'happy', 'beautiful', 'perfect', 'best', 'incredible',
            'excited', 'brilliant', 'success', 'win', 'good'
        ]
        
        negative_words = [
            'bad', 'terrible', 'awful', 'horrible', 'worst', 'hate',
            'sad', 'angry', 'disappointed', 'failure', 'lose', 'wrong',
            'problem', 'difficult', 'hard', 'struggle'
        ]
        
        emotional_words = [
            'feel', 'believe', 'think', 'know', 'love', 'hate', 'fear',
            'hope', 'dream', 'wish', 'want', 'need', 'must', 'should'
        ]
        
        text_lower = text.lower()
        words = text_lower.split()
        
        positive_count = sum(1 for w in words if w in positive_words)
        negative_count = sum(1 for w in words if w in negative_words)
        emotional_count = sum(1 for w in words if w in emotional_words)
        
        total = positive_count + negative_count
        
        if total > 0:
            sentiment_score = (positive_count - negative_count) / total
        else:
            sentiment_score = 0
        
        return {
            'sentiment_score': sentiment_score,  # -1 to 1
            'positive_count': positive_count,
            'negative_count': negative_count,
            'emotional_count': emotional_count,
            'sentiment': 'positive' if sentiment_score > 0.2 else 'negative' if sentiment_score < -0.2 else 'neutral'
        }
    
    def find_key_phrases(self, transcription: Dict, min_words: int = 3) -> List[Dict]:
        """
        Find key phrases in transcription
        
        Args:
            transcription: Transcription result
            min_words: Minimum words in a phrase
            
        Returns:
            List of key phrases with timestamps
        """
        key_phrases = []
        
        # Words that often start important phrases
        starters = ['the', 'a', 'this', 'that', 'i', 'we', 'you', 'it', 'what', 'how', 'why', 'when']
        
        # Words that often end important phrases
        enders = ['!', '?', '.', 'important', 'key', 'main', 'best', 'first', 'last']
        
        segments = transcription.get('segments', [])
        
        for segment in segments:
            text = segment['text'].strip()
            words = text.split()
            
            if len(words) >= min_words:
                # Calculate phrase importance
                importance = 0
                
                # Length factor
                importance += min(len(words), 10) * 2
                
                # Question factor
                if '?' in text:
                    importance += 10
                
                # Exclamation factor
                if '!' in text:
                    importance += 8
                
                # Keyword factor
                important_keywords = ['important', 'key', 'main', 'remember', 'note', 'listen', 'look']
                importance += sum(5 for kw in important_keywords if kw in text.lower())
                
                key_phrases.append({
                    'text': text,
                    'start': segment['start'],
                    'end': segment['end'],
                    'importance': importance,
                    'word_count': len(words)
                })
        
        # Sort by importance
        key_phrases.sort(key=lambda x: x['importance'], reverse=True)
        
        return key_phrases[:20]  # Return top 20
    
    def detect_silence(self, transcription: Dict, min_silence: float = 1.0) -> List[Dict]:
        """
        Detect silence gaps in audio
        
        Args:
            transcription: Transcription result
            min_silence: Minimum silence duration in seconds
            
        Returns:
            List of silence periods
        """
        silences = []
        segments = transcription.get('segments', [])
        
        for i in range(1, len(segments)):
            prev_end = segments[i-1]['end']
            curr_start = segments[i]['start']
            
            gap = curr_start - prev_end
            
            if gap >= min_silence:
                silences.append({
                    'start': prev_end,
                    'end': curr_start,
                    'duration': gap
                })
        
        return silences
    
    def get_speaking_rate(self, transcription: Dict) -> Dict[str, Any]:
        """
        Calculate speaking rate
        
        Args:
            transcription: Transcription result
            
        Returns:
            Dictionary with speaking rate stats
        """
        segments = transcription.get('segments', [])
        
        if not segments:
            return {'wpm': 0, 'wps': 0, 'variance': 0}
        
        rates = []
        
        for segment in segments:
            duration = segment['end'] - segment['start']
            word_count = len(segment['text'].split())
            
            if duration > 0:
                wps = word_count / duration
                wpm = wps * 60
                rates.append({
                    'start': segment['start'],
                    'wpm': wpm,
                    'wps': wps
                })
        
        wpms = [r['wpm'] for r in rates]
        
        return {
            'avg_wpm': float(np.mean(wpms)) if wpms else 0,
            'max_wpm': float(np.max(wpms)) if wpms else 0,
            'min_wpm': float(np.min(wpms)) if wpms else 0,
            'variance': float(np.var(wpms)) if len(wpms) > 1 else 0,
            'rates': rates
        }


# Test
if __name__ == "__main__":
    engine = TranscriptionEngine(model_size="small")
    print(f"Transcription Engine initialized with {engine.model_size} model")
    print(f"Device: {engine.device}")