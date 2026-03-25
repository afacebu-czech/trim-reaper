"""
Viral Content Detection Module
Uses AI to identify potentially viral moments in video content
"""

import os
import json
import re
from typing import Dict, Any, List, Optional, Tuple
from loguru import logger
from dataclasses import dataclass
import numpy as np

@dataclass
class ViralSegment:
    """A segment identified as having viral potential"""
    start: float
    end: float
    score: float
    category: str
    reason: str
    transcript: str
    confidence: float
    factors: Dict[str, float]

class ViralContentDetector:
    """
    Detect viral content moments using AI analysis
    Combines transcript analysis, visual dynamics, and engagement patterns
    """
    
    # Categories of viral content
    VIRAL_CATEGORIES = {
        'entertainment': ['funny', 'surprising', 'shocking', 'amazing', 'incredible'],
        'educational': ['learn', 'how to', 'tutorial', 'tip', 'trick', 'guide'],
        'emotional': ['inspiring', 'touching', 'moving', 'heartfelt', 'emotional'],
        'controversial': ['opinion', 'debate', 'unpopular', 'controversial'],
        'trending': ['viral', 'trending', 'popular', 'everyone', 'must see'],
        'storytelling': ['story', 'journey', 'experience', 'happened'],
        'action': ['fail', 'win', 'prank', 'challenge', 'stunt']
    }
    
    # Engagement indicators in text
    ENGAGEMENT_KEYWORDS = {
        'hooks': ['wait for it', 'watch until', 'you won\'t believe', 'plot twist', 
                  'here\'s the thing', 'let me tell you', 'this is crazy'],
        'ctas': ['follow for more', 'like and subscribe', 'share this', 'comment below',
                 'duet this', 'stitch this'],
        'questions': ['?', 'what do you think', 'guess what', 'can you'],
        'emphasis': ['!', 'literally', 'actually', 'seriously', 'honestly', 'obviously']
    }
    
    # High-value phrases
    VALUE_PHRASES = [
        'the secret', 'nobody knows', 'pro tip', 'game changer', 
        'life hack', 'you need to', 'stop doing', 'start doing',
        'this changed my life', 'game changing', 'mind blowing'
    ]
    
    def __init__(self, ai_provider: str = "openai"):
        """
        Initialize viral content detector
        
        Args:
            ai_provider: AI provider to use (openai, anthropic, local)
        """
        self.ai_provider = ai_provider
        self.ai_client = None
        
        self._init_ai_client()
    
    def _init_ai_client(self):
        """Initialize AI client"""
        try:
            if self.ai_provider == "openai":
                from openai import OpenAI
                api_key = os.environ.get("OPENAI_API_KEY")
                if api_key:
                    self.ai_client = OpenAI(api_key=api_key)
                    logger.info("OpenAI client initialized")
            elif self.ai_provider == "anthropic":
                from anthropic import Anthropic
                api_key = os.environ.get("ANTHROPIC_API_KEY")
                if api_key:
                    self.ai_client = Anthropic(api_key=api_key)
                    logger.info("Anthropic client initialized")
            elif self.ai_provider == "anthropic":
                from zai import ZaiClient
                api_key = os.environ.get("ZAI_API_KEY")
                if api_key:
                    self.ai_client = ZaiClient(api_key=api_key)
                    logger.info("ZAI client initialized")
        except ImportError as e:
            logger.warning(f"AI library not installed: {str(e)}")
        except Exception as e:
            logger.warning(f"Could not initialize AI client: {str(e)}")
    
    def detect_viral_segments(
        self, 
        analysis_data: Dict[str, Any], 
        target_duration: int = 60
    ) -> List[Dict]:
        """
        Detect viral content segments
        
        Args:
            analysis_data: Combined analysis data from transcription, detection, etc.
            target_duration: Target duration in seconds
            
        Returns:
            List of viral segment candidates
        """
        logger.info("Detecting viral content segments...")
        
        transcription = analysis_data.get('transcription', {})
        subjects = analysis_data.get('subjects', {})
        motion = analysis_data.get('motion', {})
        video_info = analysis_data.get('video_info', {})
        
        segments = transcription.get('segments', [])
        total_duration = transcription.get('duration', video_info.get('duration', 300))
        
        # Step 1: Find keyword-based hooks
        keyword_segments = self._find_keyword_hooks(segments)
        
        # Step 2: Find high-activity moments
        activity_segments = self._find_activity_peaks(subjects, motion)
        
        # Step 3: Find emotional peaks
        emotional_segments = self._find_emotional_peaks(segments)
        
        # Step 4: Find story arcs
        story_segments = self._find_story_arcs(segments)
        
        # Combine all segments
        all_candidates = []
        all_candidates.extend(keyword_segments)
        all_candidates.extend(activity_segments)
        all_candidates.extend(emotional_segments)
        all_candidates.extend(story_segments)
        
        # Use AI to refine selections if available
        if self.ai_client:
            try:
                all_candidates = self._ai_refine_segments(all_candidates, transcription, analysis_data)
            except Exception as e:
                logger.warning(f"AI refinement failed: {str(e)}")
        
        # If no segments found, create default segments
        if not all_candidates:
            all_candidates = self._create_default_segments(total_duration, target_duration)
        
        # Merge overlapping segments
        merged_segments = self._merge_overlapping(all_candidates)
        
        # Adjust to target duration
        adjusted_segments = self._adjust_durations(merged_segments, total_duration, target_duration)
        
        logger.success(f"Found {len(adjusted_segments)} viral segment candidates")
        
        return adjusted_segments
    
    def _find_keyword_hooks(self, segments: List[Dict]) -> List[Dict]:
        """Find segments with engagement keywords"""
        results = []
        
        for segment in segments:
            text = segment.get('text', '').lower()
            score = 0
            factors = {}
            
            # Check hooks
            hook_matches = sum(1 for hook in self.ENGAGEMENT_KEYWORDS['hooks'] if hook in text)
            if hook_matches > 0:
                score += hook_matches * 20
                factors['hooks'] = hook_matches * 20
            
            # Check CTAs
            cta_matches = sum(1 for cta in self.ENGAGEMENT_KEYWORDS['ctas'] if cta in text)
            if cta_matches > 0:
                score += cta_matches * 15
                factors['ctas'] = cta_matches * 15
            
            # Check questions
            if '?' in text:
                score += 10
                factors['question'] = 10
            
            # Check emphasis
            if '!' in text:
                score += 5
                factors['emphasis'] = 5
            
            # Check value phrases
            for phrase in self.VALUE_PHRASES:
                if phrase in text:
                    score += 15
                    factors['value_phrase'] = 15
                    break
            
            # Check viral keywords
            for category, keywords in self.VIRAL_CATEGORIES.items():
                keyword_matches = sum(1 for kw in keywords if kw in text)
                if keyword_matches > 0:
                    score += keyword_matches * 8
                    factors[f'{category}_keywords'] = keyword_matches * 8
            
            if score > 15:  # Threshold
                results.append({
                    'start': segment['start'],
                    'end': segment['end'],
                    'score': min(score, 100),
                    'category': 'keyword_hook',
                    'reason': f"Engaging content detected with {len(factors)} factors",
                    'transcript': segment.get('text', ''),
                    'confidence': segment.get('confidence', 0.8),
                    'factors': factors
                })
        
        return results
    
    def _find_activity_peaks(self, subjects: Dict, motion: Dict) -> List[Dict]:
        """Find moments with high visual activity"""
        results = []
        
        # Check motion data
        motion_timeline = motion.get('motion_timeline', [])
        avg_motion = motion.get('avg_motion', 0)
        
        if motion_timeline:
            # Find peaks
            for i, motion_value in enumerate(motion_timeline):
                if motion_value > avg_motion * 1.5:  # Above average
                    timestamp = i  # Assuming 1 frame per second for simplicity
                    
                    results.append({
                        'start': max(0, timestamp - 5),
                        'end': timestamp + 10,
                        'score': min(int(motion_value / avg_motion * 30), 80),
                        'category': 'activity_peak',
                        'reason': 'High visual activity detected',
                        'transcript': '',
                        'confidence': 0.7,
                        'factors': {'motion_intensity': motion_value}
                    })
        
        # Check subject detections
        subject_timeline = subjects.get('subject_timeline', [])
        if subject_timeline:
            for frame_data in subject_timeline:
                detection_count = frame_data.get('detection_count', 0)
                if detection_count >= 2:  # Multiple subjects
                    timestamp = frame_data.get('timestamp', 0)
                    
                    results.append({
                        'start': max(0, timestamp - 5),
                        'end': timestamp + 15,
                        'score': min(detection_count * 15, 70),
                        'category': 'multi_subject',
                        'reason': f'{detection_count} subjects detected in frame',
                        'transcript': '',
                        'confidence': 0.8,
                        'factors': {'subject_count': detection_count}
                    })
        
        return results
    
    def _find_emotional_peaks(self, segments: List[Dict]) -> List[Dict]:
        """Find emotionally charged moments"""
        results = []
        
        emotional_indicators = [
            'love', 'hate', 'amazing', 'terrible', 'incredible', 'awful',
            'beautiful', 'horrible', 'wonderful', 'disgusting', 'excited',
            'devastated', 'thrilled', 'heartbroken', 'ecstatic', 'furious',
            'grateful', 'disappointed', 'proud', 'ashamed', 'scared', 'brave'
        ]
        
        for segment in segments:
            text = segment.get('text', '').lower()
            
            emotion_count = sum(1 for emo in emotional_indicators if emo in text)
            
            # Check for intensity markers
            intensity = 0
            if '!' in text:
                intensity += 5
            if text.isupper():
                intensity += 10
            if '!!' in text:
                intensity += 10
            
            total_score = emotion_count * 12 + intensity
            
            if total_score > 10:
                results.append({
                    'start': segment['start'],
                    'end': segment['end'],
                    'score': min(total_score, 85),
                    'category': 'emotional',
                    'reason': 'High emotional content detected',
                    'transcript': segment.get('text', ''),
                    'confidence': 0.75,
                    'factors': {'emotions': emotion_count, 'intensity': intensity}
                })
        
        return results
    
    def _find_story_arcs(self, segments: List[Dict]) -> List[Dict]:
        """Find story arc segments"""
        results = []
        
        # Story structure indicators
        story_starters = ['so', 'once', 'when i', 'there was', 'it all started', 
                         'let me tell you', 'this is the story', 'picture this']
        story_climaxes = ['and then', 'suddenly', 'but then', 'you won\'t believe',
                         'plot twist', 'the best part', 'here\'s the thing']
        story_enders = ['and that\'s', 'the moral', 'lesson learned', 
                       'ever since', 'now i know', 'moral of the story']
        
        # Group segments into potential story arcs
        current_arc = {'start': 0, 'end': 0, 'score': 0, 'phases': []}
        
        for i, segment in enumerate(segments):
            text = segment.get('text', '').lower()
            timestamp = segment['start']
            
            # Check for story phases
            is_starter = any(s in text for s in story_starters)
            is_climax = any(c in text for c in story_climaxes)
            is_ender = any(e in text for e in story_enders)
            
            if is_starter:
                current_arc['start'] = timestamp
                current_arc['score'] += 20
                current_arc['phases'].append('start')
            
            if is_climax:
                current_arc['score'] += 30
                current_arc['phases'].append('climax')
            
            if is_ender:
                current_arc['end'] = segment['end']
                current_arc['score'] += 25
                current_arc['phases'].append('end')
                
                # Complete arc found
                if len(set(current_arc['phases'])) >= 2 and current_arc['score'] > 40:
                    results.append({
                        'start': current_arc['start'],
                        'end': current_arc['end'],
                        'score': min(current_arc['score'], 90),
                        'category': 'story',
                        'reason': 'Complete story arc detected',
                        'transcript': '',
                        'confidence': 0.8,
                        'factors': {'phases': current_arc['phases']}
                    })
                
                # Reset
                current_arc = {'start': 0, 'end': 0, 'score': 0, 'phases': []}
        
        return results
    
    def _ai_refine_segments(
        self, 
        candidates: List[Dict], 
        transcription: Dict,
        analysis_data: Dict
    ) -> List[Dict]:
        """Use AI to refine segment selection"""
        
        if not self.ai_client:
            return candidates
        
        # Prepare context for AI
        full_transcript = transcription.get('text', '')
        segments_text = "\n".join([
            f"[{s['start']:.1f}s-{s['end']:.1f}s] {s['text']}"
            for s in transcription.get('segments', [])[:100]  # Limit for token count
        ])
        
        prompt = f"""Analyze this video transcript and identify the TOP 5 most viral-worthy moments.
        
For each moment, provide:
1. Start time (in seconds)
2. End time (in seconds)  
3. A score from 1-100
4. The category (entertainment, educational, emotional, controversial, trending, storytelling, action)
5. A brief reason why it's viral-worthy

Consider:
- Engagement potential (hooks, questions, surprises)
- Emotional impact
- Action/drama level
- Educational value
- Story quality

Transcript:
{segments_text}

Respond in JSON format:
{{
    "segments": [
        {{"start": float, "end": float, "score": int, "category": "string", "reason": "string"}}
    ]
}}
"""
        
        try:
            if self.ai_provider == "openai":
                response = self.ai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a viral content expert. Analyze transcripts and identify the most engaging moments."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1000
                )
                
                content = response.choices[0].message.content
                
            elif self.ai_provider == "anthropic":
                response = self.ai_client.messages.create(
                    model="claude-3-opus-20240229",
                    max_tokens=1000,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                
                content = response.content[0].text
                
            elif self.ai_provider == "zai":
                response = self.ai_client.messages.create(
                    model="GLM-4.7-Flash",
                    max_tokens=1000,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                
                content = response.choices[0].message.content
            else:
                return candidates
            
            # Parse JSON response
            # Extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                data = json.loads(json_match.group())
                ai_segments = data.get('segments', [])
                
                # Add AI segments to candidates
                for seg in ai_segments:
                    # Get transcript for this segment
                    seg_transcript = ''
                    for s in transcription.get('segments', []):
                        if s['start'] >= seg['start'] and s['end'] <= seg['end']:
                            seg_transcript += s['text'] + ' '
                    
                    candidates.append({
                        'start': seg['start'],
                        'end': seg['end'],
                        'score': seg['score'],
                        'category': seg['category'],
                        'reason': seg['reason'],
                        'transcript': seg_transcript.strip(),
                        'confidence': 0.9,
                        'factors': {'ai_analysis': True}
                    })
            
        except Exception as e:
            logger.warning(f"AI refinement error: {str(e)}")
        
        return candidates
    
    def _create_default_segments(
        self, 
        total_duration: float, 
        target_duration: int
    ) -> List[Dict]:
        """Create default segments when none are detected"""
        segments = []
        
        # Create evenly spaced segments
        num_segments = max(1, int(total_duration / target_duration))
        segment_duration = min(target_duration, total_duration)
        
        for i in range(min(num_segments, 5)):
            start = i * (total_duration - segment_duration) / max(num_segments - 1, 1)
            end = start + segment_duration
            
            segments.append({
                'start': start,
                'end': min(end, total_duration),
                'score': 50 - i * 5,  # Decreasing scores
                'category': 'default',
                'reason': 'Default segment selection',
                'transcript': '',
                'confidence': 0.5,
                'factors': {}
            })
        
        return segments
    
    def _merge_overlapping(self, segments: List[Dict]) -> List[Dict]:
        """Merge overlapping segments"""
        if not segments:
            return []
        
        # Sort by start time
        sorted_segments = sorted(segments, key=lambda x: x['start'])
        
        merged = [sorted_segments[0]]
        
        for seg in sorted_segments[1:]:
            last = merged[-1]
            
            # Check overlap
            if seg['start'] <= last['end'] + 2:  # 2 second tolerance
                # Merge
                last['end'] = max(last['end'], seg['end'])
                last['score'] = max(last['score'], seg['score'])
                last['transcript'] += ' ' + seg['transcript']
                
                # Keep best category and reason
                if seg['score'] > last['score'] - 10:
                    last['category'] = seg['category']
                    last['reason'] = seg['reason']
            else:
                merged.append(seg)
        
        return merged
    
    def _adjust_durations(
        self, 
        segments: List[Dict], 
        total_duration: float,
        target_duration: int
    ) -> List[Dict]:
        """Adjust segment durations to target"""
        adjusted = []
        
        for seg in segments:
            current_duration = seg['end'] - seg['start']
            
            if current_duration < target_duration:
                # Extend
                extension = (target_duration - current_duration) / 2
                new_start = max(0, seg['start'] - extension)
                new_end = min(total_duration, seg['end'] + extension)
                seg['start'] = new_start
                seg['end'] = new_end
            
            elif current_duration > target_duration * 1.5:
                # Shrink (keep center)
                center = (seg['start'] + seg['end']) / 2
                half = target_duration / 2
                seg['start'] = max(0, center - half)
                seg['end'] = min(total_duration, center + half)
            
            adjusted.append(seg)
        
        # Sort by score
        adjusted.sort(key=lambda x: x['score'], reverse=True)
        
        return adjusted[:10]  # Return top 10
    
    def score_segments(self, segments: List[Dict]) -> List[Dict]:
        """
        Calculate comprehensive scores for segments
        
        Args:
            segments: List of segments to score
            
        Returns:
            Segments with updated scores
        """
        for seg in segments:
            base_score = seg.get('score', 50)
            
            # Duration bonus (30-60 seconds is ideal)
            duration = seg['end'] - seg['start']
            if 30 <= duration <= 60:
                duration_bonus = 15
            elif 20 <= duration <= 90:
                duration_bonus = 10
            else:
                duration_bonus = 5
            
            # Transcript quality bonus
            transcript = seg.get('transcript', '')
            word_count = len(transcript.split())
            transcript_bonus = min(word_count / 10, 10)
            
            # Confidence bonus
            confidence = seg.get('confidence', 0.5)
            confidence_bonus = confidence * 10
            
            # Calculate final score
            final_score = base_score + duration_bonus + transcript_bonus + confidence_bonus
            seg['score'] = min(final_score, 100)
        
        # Re-sort
        segments.sort(key=lambda x: x['score'], reverse=True)
        
        return segments
    
    def get_recommendation(self, segments: List[Dict]) -> Dict:
        """
        Get AI recommendation for best segment
        
        Args:
            segments: Scored segments
            
        Returns:
            Recommendation with reasoning
        """
        if not segments:
            return {'segment': None, 'reason': 'No segments available'}
        
        best = segments[0]
        
        return {
            'segment': best,
            'reason': f"This segment scores {best['score']:.0f}/100. {best['reason']}",
            'tips': self._get_optimization_tips(best)
        }
    
    def _get_optimization_tips(self, segment: Dict) -> List[str]:
        """Generate optimization tips for a segment"""
        tips = []
        
        duration = segment['end'] - segment['start']
        
        if duration < 30:
            tips.append("Consider extending the clip to at least 30 seconds for better engagement")
        elif duration > 90:
            tips.append("Shorter clips (30-60 seconds) often perform better on social platforms")
        
        if segment.get('category') == 'emotional':
            tips.append("Add emotional music to enhance the mood")
        
        if segment.get('category') == 'educational':
            tips.append("Consider adding on-screen text for key points")
        
        if not segment.get('transcript'):
            tips.append("Add captions to make content accessible without sound")
        
        return tips


# Test
if __name__ == "__main__":
    detector = ViralContentDetector(ai_provider="openai")
    print("Viral Content Detector initialized")