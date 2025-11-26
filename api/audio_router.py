"""
AI API Client Module
Handles communication with AI audio processing servers (Spleeter & Animal Sound Separator)
"""

import requests
import io
import soundfile as sf
import numpy as np
from typing import Optional, Tuple, List

# Server configurations
SPLEETER_URL = "http://localhost:8080"
ANIMAL_SERVER_URL = "http://localhost:5001"

SPLEETER_STEMS = ['vocals', 'drums', 'bass', 'piano', 'other']
ANIMAL_SOUNDS = ['duck', 'crow', 'owl', 'frog', 'turkey']


class AIServerClient:
    """Client for communicating with AI audio processing servers"""
    
    @staticmethod
    def check_server_health(server_url: str) -> dict:
        """Check if a server is healthy and available"""
        try:
            response = requests.get(f"{server_url}/health", timeout=5)
            if response.status_code == 200:
                return {"status": "healthy", "data": response.json()}
            return {"status": "error", "message": f"Server returned {response.status_code}"}
        except requests.exceptions.RequestException as e:
            return {"status": "error", "message": str(e)}
    
    @staticmethod
    def process_with_spleeter(
        audio_data: np.ndarray,
        sample_rate: int,
        stem: str,
        loudness: float
    ) -> Optional[Tuple[np.ndarray, int]]:
        """
        Send audio to Spleeter server for stem separation and mixing
        
        Args:
            audio_data: Audio signal as numpy array
            sample_rate: Sample rate of the audio
            stem: Which stem to adjust (vocals, drums, bass, piano, other)
            loudness: Loudness multiplier (0=mute, 1=original, 2=double)
            
        Returns:
            Tuple of (processed_audio, sample_rate) or None if failed
        """
        if stem not in SPLEETER_STEMS:
            raise ValueError(f"Invalid stem. Must be one of: {SPLEETER_STEMS}")
        
        if loudness < 0:
            raise ValueError("Loudness must be non-negative")
        
        try:
            # Convert audio to WAV format in memory
            audio_buffer = io.BytesIO()
            sf.write(audio_buffer, audio_data, sample_rate, format='WAV')
            audio_buffer.seek(0)
            
            # Prepare multipart form data
            files = {
                'file': ('input.wav', audio_buffer, 'audio/wav')
            }
            data = {
                'classifier': stem,
                'loudness': str(loudness)
            }
            
            # Send request to Spleeter server
            response = requests.post(
                f"{SPLEETER_URL}/mix",
                files=files,
                data=data,
                timeout=120  # Spleeter can take time
            )
            
            if response.status_code == 200:
                # Read the returned audio file
                result_buffer = io.BytesIO(response.content)
                processed_audio, processed_sr = sf.read(result_buffer)
                return processed_audio, processed_sr
            else:
                print(f"Spleeter server error: {response.status_code}")
                try:
                    error_detail = response.json().get('detail', 'Unknown error')
                    print(f"Error detail: {error_detail}")
                except:
                    print(f"Response: {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            return None
        except Exception as e:
            print(f"Processing error: {e}")
            return None
    
    @staticmethod
    def process_with_animal_separator(
        audio_data: np.ndarray,
        sample_rate: int,
        animal: str,
        loudness: float
    ) -> Optional[Tuple[np.ndarray, int]]:
        """
        Send audio to Animal Sound Separator server
        
        Args:
            audio_data: Audio signal as numpy array
            sample_rate: Sample rate of the audio
            animal: Which animal sound to adjust (duck, crow, owl, frog, turkey)
            loudness: Loudness multiplier (0=mute, 1=original, 2=double)
            
        Returns:
            Tuple of (processed_audio, sample_rate) or None if failed
        """
        if animal not in ANIMAL_SOUNDS:
            raise ValueError(f"Invalid animal. Must be one of: {ANIMAL_SOUNDS}")
        
        if loudness < 0:
            raise ValueError("Loudness must be non-negative")
        
        try:
            # Convert audio to WAV format in memory
            audio_buffer = io.BytesIO()
            sf.write(audio_buffer, audio_data, sample_rate, format='WAV')
            audio_buffer.seek(0)
            
            # Prepare multipart form data
            files = {
                'file': ('input.wav', audio_buffer, 'audio/wav')
            }
            data = {
                'classifier': animal,
                'loudness': str(loudness)
            }
            
            # Send request to Animal Separator server
            response = requests.post(
                f"{ANIMAL_SERVER_URL}/mix",
                files=files,
                data=data,
                timeout=120
            )
            
            if response.status_code == 200:
                # Read the returned audio file
                result_buffer = io.BytesIO(response.content)
                processed_audio, processed_sr = sf.read(result_buffer)
                return processed_audio, processed_sr
            else:
                print(f"Animal Separator server error: {response.status_code}")
                try:
                    error_detail = response.json().get('detail', 'Unknown error')
                    print(f"Error detail: {error_detail}")
                except:
                    print(f"Response: {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            return None
        except Exception as e:
            print(f"Processing error: {e}")
            return None
    
    @classmethod
    def get_available_servers(cls) -> dict:
        """Check which AI servers are available"""
        servers_status = {
            "spleeter": cls.check_server_health(SPLEETER_URL),
            "animal_separator": cls.check_server_health(ANIMAL_SERVER_URL)
        }
        return servers_status


# Convenience functions
def process_audio_with_ai(
    audio_data: np.ndarray,
    sample_rate: int,
    mode: str,
    classifier: str,
    loudness: float
) -> Optional[Tuple[np.ndarray, int]]:
    """
    Universal function to process audio with appropriate AI server
    
    Args:
        audio_data: Audio signal
        sample_rate: Sample rate
        mode: 'spleeter' or 'animal'
        classifier: Stem name or animal name
        loudness: Loudness adjustment
        
    Returns:
        Processed audio and sample rate, or None if failed
    """
    client = AIServerClient()
    
    if mode == 'spleeter':
        return client.process_with_spleeter(audio_data, sample_rate, classifier, loudness)
    elif mode == 'animal':
        return client.process_with_animal_separator(audio_data, sample_rate, classifier, loudness)
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'spleeter' or 'animal'")


def get_server_status() -> dict:
    """Get status of all AI servers"""
    return AIServerClient.get_available_servers()