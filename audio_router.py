"""
AI Audio Processor Module
Handles communication with AI audio processing servers (Spleeter & Animal Sound Separator)
"""

import requests
import io
import soundfile as sf
import numpy as np
import librosa
import json
from typing import Optional, Tuple, Dict, Union

SPLEETER_URL = "http://localhost:8080"
ANIMAL_SERVER_URL = "http://localhost:5001"

SPLEETER_STEMS = ['vocals', 'drums', 'bass', 'piano', 'other']
ANIMAL_SOUNDS = ['duck', 'cow', 'tiger', 'Cricket']


class AIAudioProcessor:
    
    @staticmethod
    def check_server_health(server_url: str, timeout: int = 3) -> Dict:
        try:
            response = requests.get(f"{server_url}/health", timeout=timeout)
            if response.status_code == 200:
                return {"status": "healthy", "data": response.json()}
            return {"status": "error", "message": f"Server returned {response.status_code}"}
        except requests.exceptions.Timeout:
            return {"status": "error", "message": "Connection timeout"}
        except requests.exceptions.ConnectionError:
            return {"status": "error", "message": "Connection refused - server may be offline"}
        except requests.exceptions.RequestException as e:
            return {"status": "error", "message": str(e)}
    
    @staticmethod
    def process_with_spleeter(
        audio_data: np.ndarray,
        sample_rate: int,
        stem: str,
        loudness: float,
        progress_callback=None
    ) -> Optional[Tuple[np.ndarray, int]]:
        if stem not in SPLEETER_STEMS:
            raise ValueError(f"Invalid stem. Must be one of: {SPLEETER_STEMS}")
        
        if loudness < 0:
            raise ValueError("Loudness must be non-negative")
        
        original_length = len(audio_data)
        original_sr = sample_rate
        
        try:
            if progress_callback:
                progress_callback(0.1, "Converting audio to WAV format...")
            
            audio_buffer = io.BytesIO()
            sf.write(audio_buffer, audio_data, sample_rate, format='WAV', subtype='PCM_16')
            audio_buffer.seek(0)
            
            if progress_callback:
                progress_callback(0.2, f"Sending to Spleeter server (adjusting {stem})...")
            
            files = {
                'file': ('input.wav', audio_buffer, 'audio/wav')
            }
            data = {
                'classifier': stem,
                'loudness': str(loudness)
            }
            
            response = requests.post(
                f"{SPLEETER_URL}/mix",
                files=files,
                data=data,
                timeout=180
            )
            
            if progress_callback:
                progress_callback(0.9, "Processing response...")
            
            if response.status_code == 200:
                result_buffer = io.BytesIO(response.content)
                processed_audio, processed_sr = sf.read(result_buffer)
                
                if processed_audio.ndim > 1:
                    processed_audio = np.mean(processed_audio, axis=1)
                
                if processed_sr != original_sr:
                    processed_audio = librosa.resample(
                        processed_audio.astype(np.float32), 
                        orig_sr=processed_sr, 
                        target_sr=original_sr
                    )
                    processed_sr = original_sr
                
                current_length = len(processed_audio)
                if current_length < original_length:
                    processed_audio = np.pad(
                        processed_audio, 
                        (0, original_length - current_length), 
                        mode='constant'
                    )
                elif current_length > original_length:
                    processed_audio = processed_audio[:original_length]
                
                if progress_callback:
                    progress_callback(1.0, "Complete!")
                
                return processed_audio, processed_sr
            else:
                error_msg = f"Server error: {response.status_code}"
                try:
                    error_detail = response.json().get('detail', 'Unknown error')
                    error_msg = f"{error_msg} - {error_detail}"
                except:
                    error_msg = f"{error_msg} - {response.text[:200]}"
                
                print(f"Spleeter error: {error_msg}")
                return None
                
        except requests.exceptions.Timeout:
            print("Request timeout - Spleeter processing took too long")
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
        adjustments: Dict[str, float],
        progress_callback=None
    ) -> Optional[Tuple[np.ndarray, int]]:
        
        # Validation
        for animal, gain in adjustments.items():
            if animal not in ANIMAL_SOUNDS:
                raise ValueError(f"Invalid animal: {animal}. Must be: {ANIMAL_SOUNDS}")
            if gain < 0:
                raise ValueError(f"Loudness for {animal} must be non-negative")
        
        original_length = len(audio_data)
        original_sr = sample_rate
        
        try:
            if progress_callback:
                progress_callback(0.1, "Converting audio to WAV format...")
            
            audio_buffer = io.BytesIO()
            sf.write(audio_buffer, audio_data, sample_rate, format='WAV', subtype='PCM_16')
            audio_buffer.seek(0)
            
            if progress_callback:
                progress_callback(0.2, f"Sending to Animal Separator (applying {len(adjustments)} adjustments)...")
            
            files = {
                'file': ('input.wav', audio_buffer, 'audio/wav')
            }
            # Send dictionary as JSON string
            data = {
                'adjustments': json.dumps(adjustments)
            }
            
            response = requests.post(
                f"{ANIMAL_SERVER_URL}/mix",
                files=files,
                data=data,
                timeout=180
            )
            
            if progress_callback:
                progress_callback(0.9, "Processing response...")
            
            if response.status_code == 200:
                result_buffer = io.BytesIO(response.content)
                processed_audio, processed_sr = sf.read(result_buffer)
                
                if processed_audio.ndim > 1:
                    processed_audio = np.mean(processed_audio, axis=1)
                
                if processed_sr != original_sr:
                    processed_audio = librosa.resample(
                        processed_audio.astype(np.float32), 
                        orig_sr=processed_sr, 
                        target_sr=original_sr
                    )
                    processed_sr = original_sr
                
                current_length = len(processed_audio)
                if current_length < original_length:
                    processed_audio = np.pad(
                        processed_audio, 
                        (0, original_length - current_length), 
                        mode='constant'
                    )
                elif current_length > original_length:
                    processed_audio = processed_audio[:original_length]
                
                if progress_callback:
                    progress_callback(1.0, "Complete!")
                
                return processed_audio, processed_sr
            else:
                error_msg = f"Server error: {response.status_code}"
                try:
                    error_detail = response.json().get('detail', 'Unknown error')
                    error_msg = f"{error_msg} - {error_detail}"
                except:
                    pass
                print(f"Animal Separator error: {error_msg}")
                return None
                
        except requests.exceptions.Timeout:
            print("Request timeout - Animal separation took too long")
            return None
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            return None
        except Exception as e:
            print(f"Processing error: {e}")
            return None
    
    @classmethod
    def get_available_servers(cls) -> Dict:
        return {
            "spleeter": cls.check_server_health(SPLEETER_URL),
            "animal_separator": cls.check_server_health(ANIMAL_SERVER_URL)
        }
    
    @classmethod
    def process_audio(
        cls,
        audio_data: np.ndarray,
        sample_rate: int,
        mode: str,
        classifier: Union[str, Dict[str, float]], 
        loudness: Optional[float] = None,
        progress_callback=None
    ) -> Optional[Tuple[np.ndarray, int]]:
        
        if mode == 'Musical Instruments Mode':
            # Spleeter expects a single string classifier and a float loudness
            if not isinstance(classifier, str) or loudness is None:
                 raise ValueError("Musical Mode requires 'classifier' (str) and 'loudness' (float)")
            return cls.process_with_spleeter(
                audio_data, sample_rate, classifier, loudness, progress_callback
            )
            
        elif mode == 'Animal Sounds Mode':
            # Animal Mode now expects a dictionary of adjustments
            if isinstance(classifier, dict):
                return cls.process_with_animal_separator(
                    audio_data, sample_rate, classifier, progress_callback
                )
            # Backward compatibility
            elif isinstance(classifier, str) and loudness is not None:
                return cls.process_with_animal_separator(
                    audio_data, sample_rate, {classifier: loudness}, progress_callback
                )
            else:
                raise ValueError("Animal Mode requires a dictionary of adjustments or classifier/loudness pair")
        else:
            raise ValueError(f"Invalid mode: {mode}")