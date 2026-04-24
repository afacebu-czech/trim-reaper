"""
Video Downloader Module
Handles video downloads from YouTube and other platforms
"""

import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List
from loguru import logger
from dataclasses import dataclass
@dataclass
class VideoFormat:
    """Represents an available video format"""
    format_id: str
    ext: str
    resolution: str
    fps: Optional[int]
    filesize: Optional[int]
    filesize_approx: Optional[int]
    vcodec: str
    acodec: str
    is_progressive: bool # Has both video and audio
    
    @property
    def filesize_mb(self) -> Optional[float]:
        """Get filesize in MB"""
        size = self.filesize or self.filesize_approx
        return round(size / (1024 * 1024), 1) if size else None
    
    @property
    def display_name(self) -> str:
        """Get display name for UI"""
        size_str = f"{self.filesize_mb}MB" if self.filesize_mb else "Unknown size"
        fps_str = f"{self.fps}fps" if self.fps else ""
        return f"{self.resolution} | {self.ext.upper()} | {size_str} {fps_str}".strip()
class VideoDownloader:
    """Download videos from various platforms including YouTube"""
    
    QUALITY_PRESETS = {
        '4K (2160p)': {
            'format': 'bestvideo[height<=2160][ext=mp4]+bestaudio[ext=m4a]/best[height<=2160][ext=mp4]/best[height<=2160]',
            'max_height': 2160,
            'description': '4K Ultra HD - Best quality, large file size'
        },
        '1440p': {
            'format': 'bestvideo[height<=1440][ext=mp4]+bestaudio[ext=m4a]/best[height<=1440][ext=mp4]/best[height<=1440]',
            'max_height': 1440,
            'description': '2K QHD - High quality'
        },
        '1080p (Full HD)': {
            'format': 'bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best[height<=1080]',
            'max_height': 1080,
            'description': 'Full HD - Recommended for most uses'
        },
        '720p (HD)': {
            'format': 'bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best[height<=720]',
            'max_height': 720,
            'description': 'HD - Good quality, smaller file'
        },
        '480p': {
            'format': 'bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/best[height<=480][ext=mp4]/best[height<=480]',
            'max_height': 480,
            'description': 'SD - Medium quality, smaller file'
        },
        '360p': {
            'format': 'bestvideo[height<=360][ext=mp4]+bestaudio[ext=m4a]/best[height<=360][ext=mp4]/best[height<=360]',
            'max_height': 360,
            'description': 'Low quality - Smallest file'
        },
        'Best Available': {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'max_height': 9999,
            'description': 'Highest quality available'
        },
        'Smallest File': {
            'format': 'worst[ext=mp4]/worst',
            'max_height': 0,
            'description': 'Lowest quality, smallest download'
        }   
    }
    
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
        
    def _get_ydl_opts(self, extra_opts: Optional[Dict]=None):
        """Get base yt-dlp options with Windows encoding fixes"""
        import sys
        
        opts = {
            'quiet': True,
            'no_warnings': True,
            'restrictfilenames': True, # ASCII-onyl filenames
        }
        
        if extra_opts:
            opts.update(extra_opts)
            
        return opts
        
    def _setup_windows_encoding(self) -> Optional[Dict]:
        """Setup UTF-8 encoding for windows subprocess calls"""
        import sys
        if sys.platform=='win32':
            old_env=os.environ.copy()
            os.environ['PYTHONIOENCODING'] = 'utf-8'
            return old_env
        return None
    
    def _restore_environment(self, old_env: Optional[Dict]):
        """Restore environment variables"""
        if old_env is not None:
            os.environ.clear()
            os.environ.update(old_env)
    
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
    
    def get_available_qualities(self, url: str) -> List[Dict[str, Any]]:
        """
        Get available quality options for a video
        
        Args:
            url: Video URL
        
        Returns:
            List of available quality options with details
        """
        try:
            import yt_dlp
            
            ydl_opts = self._get_ydl_opts({
                'extract_flat': False,
                'listformats': False
            })
              
            old_env = self._setup_windows_encoding()
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=False)
            
            finally:
                self._restore_environment(old_env)
                
            if not info:
                return []
            
            available_qualities = []
            formats = info.get('formats', [])
              
            # Get unique resolutions available
            resolutions = set()
            for fmt in formats:
                height = fmt.get('height')
                if height is not None and height > 0 and fmt.get('vcodec') != 'none':
                    resolutions.add(height)
            
            max_available = max(resolutions) if resolutions else 0
            
            for quality_name, preset in self.QUALITY_PRESETS.items():
                max_height = preset['max_height']
                
                # Check if this quality is available
                is_available = (max_height >= max_available or
                                any(r <= max_height for r in resolutions) or
                                quality_name in ['Best Available', 'Smallest File'])
                
                # Find closest matching format for size estimate
                filesize_estimate = None
                
                # Look for exact match or closest lower resolution
                for fmt in formats:
                    fmt_height = fmt.get('height') or 0 # Handle None explicitly
                    vcodec = fmt.get('vcodec') or ''
                    if fmt_height > 0 and vcodec != 'none':
                        # For "Best Available", use max resolution
                        if quality_name == 'Best Available' and fmt_height == max_available:
                            size = fmt.get('filesize') or fmt.get('filesize_approx')
                            if size:
                                filesize_estimate = round(size / (1024 * 1024), 1)
                                break
                            
                            # For specific quality, find matching resolution
                            elif fmt_height == max_height or (max_height == 0 and fmt_height < 400):
                                size = fmt.get('filesize') or fmt.get('filesize_approx')
                                if size:
                                    filesize_estimate = round(size / (1024 * 1024), 1)
                                    break
                                
                # Estimated based on duration if no exact match
                if not filesize_estimate:
                    duration = info.get('duration', 0)
                    
                    # Rough estimate: bitrate varies by resolution
                    bitrate_estimate = {
                        '4K (2160p)': 15,      # ~15 MB/min
                        '1440p': 10,
                        '1080p (Full HD)': 6,
                        '720p (HD)': 3,
                        '480p': 1.5,
                        '360p': 0.8,
                        'Best Available': 10,
                        'Smallest File': 0.5,
                    }
                    mb_per_min = bitrate_estimate.get(quality_name, 3)
                    filesize_estimate = round(duration * mb_per_min / 60, 1)
                
                available_qualities.append({
                    'name': quality_name,
                    'description': preset['description'],
                    'available': is_available,
                    'estimated_size_mb': filesize_estimate,
                    'format_selector': preset['format']
                })
                
            return available_qualities
        
        except Exception as e:
            logger.error(f"Error getting available qualities: {e}")
            
            # Return basic options on error
            return [
                {'name': 'Best Available', 'description': 'Highest quality available', 'available': True, 'estimated_size_mb': None, 'format_selector': 'best[ext=mp4]/best'},
                {'name': '720p (HD)', 'description': 'HD quality', 'available': True, 'estimated_size_mb': None, 'format_selector': 'best[height<=720][ext=mp4]/best'},
                {'name': '480p', 'description': 'SD quality', 'available': True, 'estimated_size_mb': None, 'format_selector': 'best[height<=480][ext=mp4]/best'}
            ]
        
    def get_available_formats(
        self,
        url: str
    ) -> List[VideoFormat]:
        """
        Get all available formats for a video (advanced)
        
        Args:
            url: Video URL
            
        Returns:
            List of VideoFormat objects
        """
        try:
            import yt_dlp
            
            ydl_opts = self._get_ydl_opts({
                'extract_flat': False,
            })
            
            formats = []
            old_env = self._setup_windows_encoding()
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=False)
            finally:
                self._restore_environment(old_env)
            
            if info:
                for fmt in info.get('formats', []):
                    # Skip audio-only format
                    if fmt.get('vcodec') == 'none':
                        continue
                    
                    height = fmt.get('height') or 0
                    width = fmt.get('width') or 0
                    
                    resolution = f"{width}x{height}" if width and height else "Unknown"
                    
                    formats.append(VideoFormat(
                        format_id=fmt.get('format_id', ''),
                        ext=fmt.get('ext', 'mp4'),
                        resolution=resolution,
                        fps=fmt.get('fps'),
                        filesize=fmt.get('filesize'),
                        filesize_approx=fmt.get('filesize_approx'),
                        vcodec=fmt.get('vcodec', 'unknown'),
                        acodec=fmt.get('acodec', 'unknown'),
                        is_progressive=fmt.get('acodc') != 'none'
                    ))
                        
            # Sort by resolution (highest first)
            formats.sort(key=lambda x: int(x.resolution.split('x')[-1]) if 'x' in x.resolution else 0, reverse=True)
            
            return formats
        
        except Exception as e:
            logger.error(f"Error getting formats: {e}")
            return []
        
    
    def download(
        self,
        url: str,
        quality: str = "best",
        format_id: Optional[str] = None
    ) -> Optional[str]:
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
            return self._download_youtube(url, quality, format_id)
        else:
            # Use yt-dlp for other platforms
            return self._download_ytdlp(url, quality)
    
    def _download_youtube(
        self,
        url: str,
        quality: str = "1080p (Full HD)",
        format_id: Optional[str] = None
        ) -> Optional[str]:
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
            
            # Get format selector
            if format_id:
                format_selector = f"{format_id}+bestaudio/best"
            elif quality in self.QUALITY_PRESETS:
                format_selector = self.QUALITY_PRESETS[quality]['format']
            else:
                # Default to 1080p
                format_selector = self.QUALITY_PRESETS['1080p (Full HD)']['format']
            
            # Output template - sanitize filename
            output_template = str(Path(self.output_dir) / "%(title).50s_%(id)s.%(ext)s")
            
            ydl_opts = {
                'format': format_selector,
                'outtmpl': output_template,
                'quiet': False,
                'no_warnings': False,
                'extract_audio': False,
                'merge_output_format': 'mp4',
                'overwrites': True,
                'postprocessors':[
                    {
                        'key': 'FFmpegVideoConvertor',
                        'preferedformat': 'mp4',
                    },
                    {
                        'key': 'FFmpegMetadata',
                    }
                ],
                'progress_hooks': [self._download_progress_hook],
            }
            
            logger.info(f"Downloading: {url} at quality: {quality}")
            
            info = None
            filename = None
            old_env = self._setup_windows_encoding()
            
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    if info:
                        filename = ydl.prepare_filename(info)
            except Exception as download_error:
                logger.error(f"Download error: {download_error}")
                raise
            finally:
                self._restore_environment(old_env)
            
            if filename:
                # Check if file exists with different extension
                if not os.path.exists(filename):
                    base = os.path.splitext(filename)[0]
                    for ext in ['.mp4', '.mkv', '.webm', '.avi']:
                        potential = base + ext
                        if os.path.exists(potential):
                            filename = potential
                            break
                        
                if os.path.exists(filename):
                    logger.success(f"Downloaded: {filename}")
                    return filename
            
            logger.error("Failed to download video")
            return None
            
        except ImportError:
            logger.warning("yt-dlp not installed, trying pytube")
            return self._download_pytube(url, quality)
        except Exception as e:
            logger.error(f"Download error: {str(e)}")
            return self._download_pytube(url, quality)
        
    def _download_progress_hook(
        self,
        d: Dict
        ):
        """Progress hook for download status"""
        if d['status'] == 'downloading':
            # Calculate progress
            downloaded = d.get('downloaded_bytes', 0)
            total = d.get('total_bytes') or d.get('total_bytes_estimate', 0)
            
            if total > 0:
                percent = (downloaded / total) * 100
                speed = d.get('speed', 0)
                speed_mb = speed / (1024 * 1024) if speed else 0
                
                logger.info(f"Progress: {percent:.1f}% | Speed: {speed_mb:.1f} MB/s")
    
    def _download_pytube(
        self,
        url: str,
        quality: str = "1080p (Full HD)"
        ) -> Optional[str]:
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
            
            # Map quality to pytube resolution
            resolution_map = {
                '4K (2160p)': '2160p',
                '1440p': '1440p',
                '1080p (Full HD)': '1080p',
                '720p (HD)': '720p',
                '480p': '480p',
                '360p': '360p'
            }
            
            target_res = resolution_map.get(quality, '720p')
            
            # Try to get the target resolution
            stream = yt.streams.filter(
                progressive=True,
                file_extension='mp4',
                resolution=target_res
            ).first()
            
            # Fallback to best available
            if not stream:
                stream = yt.streams.filter(
                    progressive=True,
                    file_extension='mp4'
                ).order_by('resolution').desc().first()
                
            if stream:
                output_path = stream.download(output_path=self.output_dir)
                logger.success(f"Downloaded with pytube: {output_path}")
                return output_path
            
            return None
            
        except ImportError:
            logger.error("pytube not installed")
            return None
        except Exception as e:
            logger.error(f"Pytube error: {str(e)}")
            return None
    
    def _download_ytdlp(
        self,
        url: str,
        quality: str = "best") -> Optional[str]:
        """
        Generic download using yt-dlp for non-Youtube platforms
        
        Args:
            url: Video URL
            quality: Video quality
            
        Returns:
            Path to downloaded video
        """
        try:
            import yt_dlp
            
            output_template = str(Path(self.output_dir) / "video_%(id)s.%(ext)s")
            
            # Simple format for other platforms
            if quality in self.QUALITY_PRESETS:
                format_selector = self.QUALITY_PRESETS[quality]['format']
            else:
                format_selector = 'best[ext=mp4]/best'
            
            ydl_opts = self._get_ydl_opts({
                'format': format_selector,
                'outtmpl': output_template,
                'merge_output_format': 'mp4',
                'overwrites': True,
            })
            
            old_env = self._setup_windows_encoding()
            
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    
                    if info:
                        filename = ydl.prepare_filename(info)
                        if os.path.exists(filename):
                            return filename
            finally:
                self._restore_environment(old_env)
            
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
        Get detailed video information without downloading
        
        Args:
            url: Video URL
            
        Returns:
            Dictionary with video information
        """
        try:
            import yt_dlp
            
            ydl_opts = self._get_ydl_opts({
                'extract_flat': False,
            })
            
            old_env = self._setup_windows_encoding()
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=False)
            finally:
                self._restore_environment(old_env)
                
            if info:
                # Get available resolutions
                formats = info.get('formats', [])
                resolutions = set()
                for fmt in formats:
                    height = fmt.get('height')
                    vcodec = fmt.get('vcodec') or ''
                    
                    # Ensure height is a valid positive integer
                    if height is not None and isinstance(height, (int, float)) and int(height)>0:
                        if vcodec != 'none':
                            resolutions.add(int(height))
                            
                # Calculate file size estimates
                duration = info.get('duration', 0)
                
                return {
                    'title': info.get('title', 'Unknown'),
                    'duration': info.get('duration', 0),
                    'duration_formatted': self._format_duration(info.get('duration', 0)),
                    'description': info.get('description', '')[:500] if info.get('description') else '',
                    'uploader': info.get('uploader', 'Unknown'),
                    'channel': info.get('channel', info.get('uploader', 'Unknown')),
                    'view_count': info.get('view_count', 0),
                    'like_count': info.get('like_count', 0),
                    'thumbnail': info.get('thumbnail', ''),
                    'width': info.get('width', 0),
                    'height': info.get('height', 0),
                    'fps': info.get('fps', 0),
                    'available_resolutions': sorted(list(resolutions), reverse=True),
                    'max_resolution': max(resolutions) if resolutions else 0,
                    'platform': self.detect_platform(url),
                    'url': url,
                }
                
            return {}
        
        except Exception as e:
            logger.error(f"Error getting video info: {e}")
            return {'error': str(e)}
        
    def _format_duration(self, seconds: int) -> str:
        """Format duration in seconds to HH:MM:SS"""
        if seconds is None:
            return "0:00"
        seconds = int(seconds or 0)
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        
        if hours > 0:
            return f"{hours}:{minutes:02d} : {secs:02d}"
        return f"{minutes}:{secs:02d}"
    
    def cleanup(self):
        """Clean up temporary downloads"""
        import shutil
        
        if os.path.exists(self.output_dir):
            try:
                shutil.rmtree(self.output_dir)
                Path(self.output_dir).mkdir(parents=True, exist_ok=True)
                logger.info("Cleaned up download directory")
            except Exception as e:
                logger.warning(f"Could not clean up: {str(e)}")


# Test 
if __name__ == "__main__":
    downloader = VideoDownloader()
    
    # Test with a sample YouTube URL
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    
    print("Getting video info...")
    info = downloader.get_video_info(test_url)
    
    print(f"Title: {info.get('title')}")
    print(f"Duration: {info.get('duration_formatted')}")
    print(f"Max Resolution: {info.get('max_resolution')}p")
    print(f"Available: {info.get('available_resolutions')}")
    
    print("\nAvailable qualities:")
    qualities = downloader.get_available_qualities(test_url)
    for q in qualities:
        print(f"    - {q['name']} : {q['description']} (Available: {q['available']})")