"""
AI Integration Module
Provides unified interface for multiple AI providers (OpenAI, Anthropic, Local)
"""

import os
import json
import re
from typing import Dict, Any, List, Optional, Tuple
from loguru import logger
from abc import ABC, abstractmethod

class AIProvider(ABC):
    """Abstract base class for AI providers"""
    
    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Generate text from prompt"""
        pass
    
    @abstractmethod
    def analyze_content(self, content: str, task: str) -> Dict[str, Any]:
        """Analyze content for specific task"""
        pass

class OpenAIProvider(AIProvider):
    """OpenAI GPT integration"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.client = None
        
        if self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
                logger.info("OpenAI client initialized")
            except ImportError:
                logger.warning("openai package not installed")
    
    def generate(
        self, 
        prompt: str, 
        system_prompt: str = "",
        model: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> str:
        """Generate text using OpenAI"""
        
        if not self.client:
            return self._fallback_generate(prompt)
        
        try:
            messages = []
            
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI generation error: {str(e)}")
            return self._fallback_generate(prompt)
    
    def analyze_content(self, content: str, task: str) -> Dict[str, Any]:
        """Analyze content using OpenAI"""
        
        prompt = f"""Analyze the following content for {task}.

Content:
{content}

Provide your analysis in JSON format with the following structure:
{{
    "analysis": "detailed analysis",
    "score": <0-100>,
    "highlights": ["point1", "point2", ...],
    "recommendations": ["rec1", "rec2", ...]
}}
"""
        
        result = self.generate(prompt, system_prompt="You are an expert content analyst.")
        
        try:
            # Extract JSON
            json_match = re.search(r'\{[\s\S]*\}', result)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        return {
            "analysis": result,
            "score": 50,
            "highlights": [],
            "recommendations": []
        }
    
    def _fallback_generate(self, prompt: str) -> str:
        """Fallback when API not available"""
        return json.dumps({
            "analysis": "AI analysis unavailable - API key not configured",
            "score": 50,
            "highlights": ["Configure API key for full analysis"],
            "recommendations": ["Add your API key in the settings"]
        })

class AnthropicProvider(AIProvider):
    """Anthropic Claude integration"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.client = None
        
        if self.api_key:
            try:
                from anthropic import Anthropic
                self.client = Anthropic(api_key=self.api_key)
                logger.info("Anthropic client initialized")
            except ImportError:
                logger.warning("anthropic package not installed")
    
    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        model: str = "claude-3-opus-20240229",
        max_tokens: int = 2000
    ) -> str:
        """Generate text using Anthropic"""
        
        if not self.client:
            return self._fallback_generate(prompt)
        
        try:
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            
            response = self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "user", "content": full_prompt}
                ]
            )
            
            return response.content[0].text
            
        except Exception as e:
            logger.error(f"Anthropic generation error: {str(e)}")
            return self._fallback_generate(prompt)
    
    def analyze_content(self, content: str, task: str) -> Dict[str, Any]:
        """Analyze content using Anthropic"""
        
        prompt = f"""Analyze the following content for {task}.

Content:
{content}

Provide your analysis in JSON format.
"""
        
        result = self.generate(prompt, system_prompt="You are an expert content analyst. Always respond in valid JSON format.")
        
        try:
            json_match = re.search(r'\{[\s\S]*\}', result)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        return {
            "analysis": result,
            "score": 50,
            "highlights": [],
            "recommendations": []
        }
    
    def _fallback_generate(self, prompt: str) -> str:
        """Fallback when API not available"""
        return json.dumps({
            "analysis": "AI analysis unavailable - API key not configured",
            "score": 50,
            "highlights": ["Configure API key for full analysis"],
            "recommendations": ["Add your API key in the settings"]
        })

class LocalProvider(AIProvider):
    """Local model provider (rule-based fallback)"""
    
    def __init__(self):
        logger.info("Local provider initialized (rule-based)")
    
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Generate using local rules"""
        
        # Analyze the prompt and provide relevant response
        if "viral" in prompt.lower() or "engagement" in prompt.lower():
            return self._analyze_viral_potential(prompt)
        elif "transcript" in prompt.lower():
            return self._analyze_transcript(prompt)
        elif "segment" in prompt.lower():
            return self._suggest_segments(prompt)
        else:
            return self._general_analysis(prompt)
    
    def analyze_content(self, content: str, task: str) -> Dict[str, Any]:
        """Analyze content using local rules"""
        
        if "viral" in task.lower():
            return self._viral_analysis(content)
        elif "sentiment" in task.lower():
            return self._sentiment_analysis(content)
        else:
            return {
                "analysis": "Local analysis performed",
                "score": self._calculate_score(content),
                "highlights": self._extract_highlights(content),
                "recommendations": self._generate_recommendations(content)
            }
    
    def _analyze_viral_potential(self, content: str) -> str:
        """Analyze viral potential"""
        
        viral_keywords = [
            'amazing', 'incredible', 'shocking', 'secret', 'nobody knows',
            'must see', 'viral', 'trending', 'insane', 'crazy'
        ]
        
        content_lower = content.lower()
        matches = sum(1 for kw in viral_keywords if kw in content_lower)
        
        score = min(matches * 10 + 30, 100)
        
        return json.dumps({
            "viral_score": score,
            "factors": {
                "keyword_density": matches,
                "emotional_markers": content.count('!'),
                "question_hooks": content.count('?')
            },
            "recommendations": [
                "Add more emotional hooks" if matches < 3 else "Good emotional content",
                "Include a call-to-action",
                "Consider adding trending hashtags"
            ]
        })
    
    def _analyze_transcript(self, content: str) -> str:
        """Analyze transcript content"""
        
        words = content.split()
        word_count = len(words)
        
        # Find potential hooks
        hook_patterns = [
            r'\b(wait for it)\b',
            r'\b(you won\'t believe)\b',
            r'\b(here\'s the thing)\b',
            r'\b(let me tell you)\b',
            r'\b(this is crazy)\b'
        ]
        
        hooks_found = []
        for pattern in hook_patterns:
            matches = re.findall(pattern, content.lower())
            hooks_found.extend(matches)
        
        return json.dumps({
            "word_count": word_count,
            "estimated_duration": word_count / 150,  # ~150 WPM average
            "hooks_found": hooks_found,
            "questions": content.count('?'),
            "emphasis": content.count('!'),
            "engagement_potential": "high" if len(hooks_found) > 2 else "medium" if len(hooks_found) > 0 else "low"
        })
    
    def _suggest_segments(self, content: str) -> str:
        """Suggest content segments"""
        
        return json.dumps({
            "segments": [
                {
                    "start": 0,
                    "end": 30,
                    "type": "hook",
                    "reason": "Opening segment typically has highest engagement"
                },
                {
                    "start": 30,
                    "end": 60,
                    "type": "content",
                    "reason": "Main content delivery"
                },
                {
                    "start": 60,
                    "end": 90,
                    "type": "climax",
                    "reason": "Build to peak moment"
                }
            ]
        })
    
    def _general_analysis(self, content: str) -> str:
        """General content analysis"""
        
        return json.dumps({
            "length": len(content),
            "word_count": len(content.split()),
            "complexity": "medium",
            "readability_score": 60
        })
    
    def _viral_analysis(self, content: str) -> Dict[str, Any]:
        """Detailed viral analysis"""
        
        viral_keywords = [
            'amazing', 'incredible', 'shocking', 'secret', 'nobody knows',
            'must see', 'viral', 'trending', 'insane', 'crazy', 'unbelievable'
        ]
        
        content_lower = content.lower()
        keyword_matches = [kw for kw in viral_keywords if kw in content_lower]
        
        score = min(len(keyword_matches) * 8 + content.count('!') * 2 + 30, 100)
        
        return {
            "analysis": f"Found {len(keyword_matches)} viral keywords and {content.count('!')} emphasis markers",
            "score": score,
            "highlights": keyword_matches,
            "recommendations": [
                "Add more emotional triggers" if len(keyword_matches) < 3 else "Strong viral potential",
                "Include a clear call-to-action",
                "Consider timing for maximum engagement"
            ]
        }
    
    def _sentiment_analysis(self, content: str) -> Dict[str, Any]:
        """Simple sentiment analysis"""
        
        positive_words = ['good', 'great', 'amazing', 'love', 'excellent', 'wonderful', 'fantastic']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'horrible', 'worst']
        
        content_lower = content.lower()
        positive_count = sum(1 for w in positive_words if w in content_lower)
        negative_count = sum(1 for w in negative_words if w in content_lower)
        
        if positive_count > negative_count:
            sentiment = "positive"
            score = 60 + positive_count * 5
        elif negative_count > positive_count:
            sentiment = "negative"
            score = 40 - negative_count * 5
        else:
            sentiment = "neutral"
            score = 50
        
        return {
            "analysis": f"Sentiment: {sentiment}",
            "score": max(0, min(100, score)),
            "highlights": [f"Positive words: {positive_count}", f"Negative words: {negative_count}"],
            "recommendations": ["Content has balanced sentiment"] if sentiment == "neutral" else [f"Content leans {sentiment}"]
        }
    
    def _calculate_score(self, content: str) -> int:
        """Calculate engagement score"""
        
        score = 50
        
        # Length factor
        word_count = len(content.split())
        if 50 <= word_count <= 200:
            score += 10
        
        # Punctuation factor
        score += min(content.count('!') * 2, 10)
        score += min(content.count('?') * 3, 15)
        
        # Keyword factor
        engagement_words = ['you', 'your', 'this', 'here', 'now']
        for word in engagement_words:
            if word in content.lower():
                score += 3
        
        return min(score, 100)
    
    def _extract_highlights(self, content: str) -> List[str]:
        """Extract key highlights"""
        
        highlights = []
        
        # Find sentences with emphasis
        sentences = content.split('.')
        for sentence in sentences:
            if '!' in sentence or '?' in sentence:
                highlights.append(sentence.strip())
        
        return highlights[:5]
    
    def _generate_recommendations(self, content: str) -> List[str]:
        """Generate content recommendations"""
        
        recommendations = []
        
        if len(content.split()) < 50:
            recommendations.append("Consider adding more content for better engagement")
        
        if content.count('?') < 2:
            recommendations.append("Add questions to increase engagement")
        
        if content.count('!') < 2:
            recommendations.append("Add emphasis markers for stronger impact")
        
        if not recommendations:
            recommendations.append("Content looks good for engagement!")
        
        return recommendations

class AIIntegration:
    """
    Unified AI integration supporting multiple providers
    """
    
    def __init__(self, provider: str = "openai"):
        """
        Initialize AI integration
        
        Args:
            provider: AI provider name (openai, anthropic, local)
        """
        self.provider_name = provider
        self.provider = self._init_provider(provider)
    
    def _init_provider(self, provider: str) -> AIProvider:
        """Initialize the specified provider"""
        
        if provider == "openai":
            return OpenAIProvider()
        elif provider == "anthropic":
            return AnthropicProvider()
        else:
            return LocalProvider()
    
    def switch_provider(self, provider: str):
        """Switch to a different provider"""
        self.provider_name = provider
        self.provider = self._init_provider(provider)
        logger.info(f"Switched to {provider} provider")
    
    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        **kwargs
    ) -> str:
        """Generate text using current provider"""
        return self.provider.generate(prompt, system_prompt, **kwargs)
    
    def analyze_content(
        self,
        content: str,
        task: str = "general"
    ) -> Dict[str, Any]:
        """Analyze content using current provider"""
        return self.provider.analyze_content(content, task)
    
    def find_viral_moments(
        self,
        transcript: str,
        duration: float,
        target_duration: int = 60
    ) -> List[Dict]:
        """
        Find viral moments in transcript
        
        Args:
            transcript: Full transcript text
            duration: Total video duration
            target_duration: Target clip duration
            
        Returns:
            List of suggested viral moments
        """
        
        prompt = f"""Analyze this video transcript and identify the TOP 5 most viral-worthy moments.

Video duration: {duration} seconds
Target clip duration: {target_duration} seconds

Transcript:
{transcript[:5000]}  # Limit for token count

For each moment, provide:
1. Start time (in seconds)
2. End time (in seconds)
3. Score (1-100)
4. Category
5. Brief reason

Respond in JSON format:
{{
    "moments": [
        {{"start": float, "end": float, "score": int, "category": "string", "reason": "string"}}
    ]
}}
"""
        
        result = self.generate(prompt, system_prompt="You are a viral content expert. Always respond in valid JSON.")
        
        try:
            json_match = re.search(r'\{[\s\S]*\}', result)
            if json_match:
                data = json.loads(json_match.group())
                return data.get('moments', [])
        except:
            pass
        
        # Fallback
        return self._generate_default_moments(duration, target_duration)
    
    def _generate_default_moments(
        self,
        duration: float,
        target_duration: int
    ) -> List[Dict]:
        """Generate default moments when AI fails"""
        
        moments = []
        num_segments = min(5, int(duration / target_duration))
        
        for i in range(num_segments):
            start = i * (duration - target_duration) / max(num_segments - 1, 1)
            end = start + target_duration
            
            moments.append({
                "start": start,
                "end": min(end, duration),
                "score": 70 - i * 5,
                "category": "general",
                "reason": f"Segment {i + 1} of video"
            })
        
        return moments
    
    def generate_title(self, content: str) -> str:
        """Generate an engaging title for content"""
        
        prompt = f"Generate a catchy, viral-worthy title for this content: {content[:500]}"
        
        title = self.generate(prompt, max_tokens=50)
        
        # Clean up
        title = title.strip().strip('"').strip("'")
        
        return title
    
    def generate_description(self, content: str) -> str:
        """Generate a description for content"""
        
        prompt = f"Write a compelling social media description for this content: {content[:1000]}"
        
        description = self.generate(prompt, max_tokens=200)
        
        return description.strip()
    
    def suggest_hashtags(self, content: str, count: int = 10) -> List[str]:
        """Suggest hashtags for content"""
        
        prompt = f"Suggest {count} relevant hashtags for this content: {content[:500]}. Return as a JSON array."
        
        result = self.generate(prompt, max_tokens=100)
        
        try:
            json_match = re.search(r'\[[\s\S]*?\]', result)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        # Fallback hashtags
        return ["#viral", "#trending", "#fyp", "#reels", "#content"]


# Test
if __name__ == "__main__":
    # Test local provider
    ai = AIIntegration(provider="local")
    print("AI Integration initialized with local provider")
    
    result = ai.analyze_content("This is an amazing video! You won't believe what happens next!", "viral")
    print(f"Analysis: {result}")