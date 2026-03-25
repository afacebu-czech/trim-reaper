"""
Video Downloader Module
Handles video downloads from YouTube and other platforms
"""

import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any
from loguru import logger

class VideoDownloader:
    """Download videos from various platforms including YouTube"""
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize the video downloader
        
        Args:
            output_dir: Directory to save downloaded videos
        """
        self.output_dir = output_dir or str(Path(tempfile.gettempdir()) / "reels_downloads")
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Platform patterns
        self.platform_patterns = {
            'youtube': [
                r'(youtube\.com/watch\?v=)',
                r'(youtu\.be/)',
                r'(youtube\.com/shorts/)'
            ],
            'tiktok': [
                r'(tiktok\.com/)',
                r'(vm\.tiktok\.com/)'
            ],
            'instagram': [
                r'(instagram\.com/reel/)',
                r'(instagram\.com/p/)'
            ],
            'twitter': [
                r'(twitter\.com/)',
                r'(x\.com/)'
            ],
            'vimeo': [
                r'(vimeo\.com/)'
            ],
            'facebook': [
                r'(facebook\.com/.*videos)',
                r'(fb\.watch/)'
            ]
        }
    
    def detect_platform(self, url: str) -> Optional[str]:
        """
        Detect the platform from URL
        
        Args:
            url: Video URL
            
        Returns:
            Platform name or None
        """
        url_lower = url.lower()
        for platform, patterns in self.platform_patterns.items():
            for pattern in patterns:
                if re.search(pattern, url_lower):
                    return platform
        return None
    
    def download(self, url: str, quality: str = "best") -> Optional[str]:
        """
        Download video from URL
        
        Args:
            url: Video URL
            quality: Video quality preference (best, medium, worst)
            
        Returns:
            Path to downloaded video or None on failure
        """
        platform = self.detect_platform(url)
        
        if platform == 'youtube':
            return self._download_youtube(url, quality)
        else:
            # Use yt-dlp for other platforms
            return self._download_ytdlp(url, quality)
    
    def _download_youtube(self, url: str, quality: str = "best") -> Optional[str]:
        """
        Download YouTube video using yt-dlp
        
        Args:
            url: YouTube URL
            quality: Video quality
            
        Returns:
            Path to downloaded video
        """
        try:
            import yt_dlp
            
            # Configure quality format
            if quality == "best":
                format_selector = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"
            elif quality == "medium":
                format_selector = "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best[height<=720]"
            else:
                format_selector = "worst[ext=mp4]/worst"
            
            # Output template
            output_template = str(Path(self.output_dir) / "%(title)s_%(id)s.%(ext)s")
            
            ydl_opts = {
                'format': format_selector,
                'outtmpl': output_template,
                'quiet': True,
                'no_warnings': True,
                'extract_audio': False,
                'merge_output_format': 'mp4',
                'postprocessors': [{
                    'key': 'FFmpegVideoConvertor',
                    'preferedformat': 'mp4',
                }],
            }
            
            logger.info(f"Downloading video from: {url}")
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                
                if info:
                    # Get the downloaded file path
                    filename = ydl.prepare_filename(info)
                    
                    # Check if file exists with different extension
                    if not os.path.exists(filename):
                        base = os.path.splitext(filename)[0]
                        for ext in ['.mp4', '.mkv', '.webm', '.avi']:
                            potential = base + ext
                            if os.path.exists(potential):
                                filename = potential
                                break
                    
                    if os.path.exists(filename):
                        logger.success(f"Video downloaded: {filename}")
                        return filename
            
            logger.error("Failed to download video")
            return None
            
        except ImportError:
            logger.warning("yt-dlp not installed, trying pytube")
            return self._download_pytube(url, quality)
        except Exception as e:
            logger.error(f"Error downloading YouTube video: {str(e)}")
            return self._download_pytube(url, quality)
    
    def _download_pytube(self, url: str, quality: str = "best") -> Optional[str]:
        """
        Fallback download using pytube
        
        Args:
            url: YouTube URL
            quality: Video quality
            
        Returns:
            Path to downloaded video
        """
        try:
            from pytube import YouTube
            
            yt = YouTube(url)
            
            # Get video stream
            if quality == "best":
                stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
            elif quality == "medium":
                stream = yt.streams.filter(progressive=True, file_extension='mp4', resolution='720p').first()
                if not stream:
                    stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
            else:
                stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').asc().first()
            
            if stream:
                output_path = stream.download(output_path=self.output_dir)
                logger.success(f"Video downloaded with pytube: {output_path}")
                return output_path
            
            return None
            
        except ImportError:
            logger.error("pytube not installed")
            return None
        except Exception as e:
            logger.error(f"Error with pytube download: {str(e)}")
            return None
    
    def _download_ytdlp(self, url: str, quality: str = "best") -> Optional[str]:
        """
        Generic download using yt-dlp
        
        Args:
            url: Video URL
            quality: Video quality
            
        Returns:
            Path to downloaded video
        """
        try:
            import yt_dlp
            
            output_template = str(Path(self.output_dir) / "video_%(id)s.%(ext)s")
            
            ydl_opts = {
                'format': 'best[ext=mp4]/best',
                'outtmpl': output_template,
                'quiet': True,
                'no_warnings': True,
                'merge_output_format': 'mp4',
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                
                if info:
                    filename = ydl.prepare_filename(info)
                    if os.path.exists(filename):
                        return filename
            
            return None
            
        except Exception as e:
            logger.error(f"Error downloading video: {str(e)}")
            return None
    
    def get_video_info(self, url: str) -> Dict[str, Any]:
        """
        Get video information without downloading
        
        Args:
            url: Video URL
            
        Returns:
            Dictionary with video information
        """
        try:
            import yt_dlp
            
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                
                if info:
                    return {
                        'title': info.get('title', 'Unknown'),
                        'duration': info.get('duration', 0),
                        'description': info.get('description', ''),
                        'uploader': info.get('uploader', 'Unknown'),
                        'view_count': info.get('view_count', 0),
                        'like_count': info.get('like_count', 0),
                        'thumbnail': info.get('thumbnail', ''),
                        'width': info.get('width', 0),
                        'height': info.get('height', 0),
                        'fps': info.get('fps', 0),
                    }
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting video info: {str(e)}")
            return {}
    
    def cleanup(self):
        """Clean up temporary downloads"""
        import shutil
        
        if os.path.exists(self.output_dir):
            try:
                shutil.rmtree(self.output_dir)
                logger.info("Cleaned up download directory")
            except Exception as e:
                logger.warning(f"Could not clean up: {str(e)}")


# Test function
if __name__ == "__main__":
    downloader = VideoDownloader()
    
    # Test with a sample YouTube URL
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    
    info = downloader.get_video_info(test_url)
    print(f"Video Info: {info}")