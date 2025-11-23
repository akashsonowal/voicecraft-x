#!/usr/bin/env python3
"""
AkaalTTSMetrics: Comprehensive TTS evaluation metrics class.

This class combines functionality from evaluate_model.py and evaluate_tts_with_reference.py:
1. Generates reference audio using Bhashini TTS API OR accepts existing reference audio paths directly
2. Evaluates XTTS checkpoints with multiple metrics:
   - Latency and RTF (Real-Time Factor)
   - Speaker similarity (zero-shot cloning)
   - MOS (Mean Opinion Score)
   - PESQ (Perceptual Evaluation of Speech Quality)
   - MCD (Mel Cepstral Distortion)
   - Model size

Example usage:
    # Initialize metrics class
    metrics = AkaalTTSMetrics(device="cuda:0", api_key="your-api-key")
    
    # Option 1: Generate reference audio files from Bhashini API
    reference_audios = metrics.generate_all_reference_audios(
        output_dir=Path("reference_audio"),
        languages=["hi", "en", "bn"]
    )
    
    # Option 2: Use existing reference audio files directly
    reference_audios = metrics.generate_all_reference_audios(
        existing_reference_audios={
            "hi": Path("reference_audio/reference_hi.wav"),
            "en": Path("reference_audio/reference_en.wav"),
            "bn": Path("reference_audio/reference_bn.wav"),
        },
        languages=["hi", "en", "bn"]
    )
    
    # Option 3: Pass reference audio dictionary directly to evaluate_checkpoints
    results = metrics.evaluate_checkpoints(
        checkpoint_dir=Path("checkpoints"),
        speaker_audio=Path("speaker.wav"),
        reference_audios={
            "hi": Path("reference_audio/reference_hi.wav"),
            "en": Path("reference_audio/reference_en.wav"),
            "bn": Path("reference_audio/reference_bn.wav"),
        },
        output_dir=Path("evaluation_results"),
        languages=["hi", "en", "bn"],
        per_language_reference=True  # Use language-specific reference for each language
    )
    
    # Option 4: Pass reference audio directory directly (auto-loads reference_{lang}.wav files)
    results = metrics.evaluate_checkpoints(
        checkpoint_dir=Path("checkpoints"),
        speaker_audio=Path("speaker.wav"),
        reference_audio_dir=Path("reference_audio"),  # Directory containing reference_hi.wav, reference_en.wav, etc.
        output_dir=Path("evaluation_results"),
        languages=["hi", "en", "bn"],
        per_language_reference=True
    )
    
    # Or evaluate without reference audio (no PESQ/MCD)
    results = metrics.evaluate_checkpoints(
        checkpoint_dir=Path("checkpoints"),
        speaker_audio=Path("speaker.wav"),
        output_dir=Path("evaluation_results"),
        languages=["hi", "en", "bn"]
    )
"""

import base64
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
import torch
import torchaudio
from tqdm import tqdm

try:
    from underthesea import sent_tokenize
except ImportError:
    def sent_tokenize(text: str) -> List[str]:
        return text.split(".")

try:
    from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
    PESQ_AVAILABLE = True
except ImportError:
    PESQ_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts


TARGET_SAMPLE_RATE = 24000

# Language codes
DEFAULT_LANGUAGES = [
    "en", "hi", "bn", "bho", "chha", "kn", "mag", "mr", "mai", "te", "guj"
]

# Mapping from XTTS language codes to Bhashini API language codes
BHASHINI_LANG_MAP = {
    "hi": "hi",  # Hindi
    "bn": "bn",  # Bengali
    "bho": "bho",  # Bhojpuri
    "chha": "chha",  # Chhattisgarhi
    "kn": "kn",  # Kannada
    "mag": "mag",  # Magahi
    "mai": "mai",  # Maithili
    "mr": "mr",  # Marathi
    "te": "te",  # Telugu
    "guj": "gu",  # Gujarati (API uses 'gu')
    "en": "en",  # English
}

# Evaluation texts for pregnant to-be-mothers in low income communities
DEFAULT_PROMPTS: Dict[str, str] = {
    "en": "Rest well and avoid heavy work. Take your iron and folic acid tablets daily.",
    "hi": "अच्छी तरह आराम करें और भारी काम से बचें। रोजाना अपनी आयरन और फोलिक एसिड की गोलियां लें।",
    "bn": "ভালোভাবে বিশ্রাম নিন এবং ভারী কাজ এড়িয়ে চলুন। প্রতিদিন আপনার আয়রন এবং ফোলিক অ্যাসিড ট্যাবলেট গ্রহণ করুন।",
    "bho": "अच्छा आराम करीं आ भारी काम से बचीं। रोज अपनी आयरन आ फोलिक एसिड के गोली लीं।",
    "chha": "अच्छा आराम करव आ भारी काम से बचव। रोज अपनी आयरन आ फोलिक एसिड के गोली लेव।",
    "kn": "ಚೆನ್ನಾಗಿ ವಿಶ್ರಾಂತಿ ಪಡೆಯಿರಿ ಮತ್ತು ಭಾರೀ ಕೆಲಸವನ್ನು ತಪ್ಪಿಸಿ। ಪ್ರತಿದಿನ ನಿಮ್ಮ ಕಬ್ಬಿಣ ಮತ್ತು ಫೋಲಿಕ್ ಆಮ್ಲದ ಮಾತ್ರೆಗಳನ್ನು ತೆಗೆದುಕೊಳ್ಳಿ।",
    "mag": "अच्छा आराम करीं आ भारी काम से बचीं। रोज अपनी आयरन आ फोलिक एसिड के गोली लीं।",
    "mai": "अच्छा आराम करूं आ भारी काम से बचूं। रोज अपनी आयरन आ फोलिक एसिड के गोली लूं।",
    "mr": "चांगले विश्रांती घ्या आणि जड काम टाळा। दररोज आपल्या लोह आणि फोलिक ऍसिड गोळ्या घ्या।",
    "te": "బాగా విశ్రాంతి తీసుకోండి మరియు భారీ పనిని నివారించండి। ప్రతిరోజు మీ ఇనుము మరియు ఫోలిక్ యాసిడ్ మాత్రలను తీసుకోండి।",
    "guj": "સારી રીતે આરામ કરો અને ભારે કામથી બચો। દરરોજ તમારી આયર્ન અને ફોલિક એસિડની ગોળીઓ લો।",
}


class AkaalTTSMetrics:
    """
    Comprehensive TTS evaluation metrics class.
    
    Combines reference audio generation (Bhashini API) and model evaluation
    (XTTS checkpoints) with multiple quality metrics.
    """
    
    def __init__(
        self,
        device: Optional[str] = None,
        api_key: Optional[str] = None,
        default_api_key: str = "5Yknj5sVdr1hA4V7JoKrzISnVfnXl5iUCnbZ9NZTbUrcudoAOzq0fc6VzQF19iB9",
    ):
        """
        Initialize AkaalTTSMetrics.
        
        Args:
            device: Device to use (e.g., 'cuda:0'). Auto-detects if None.
            api_key: Bhashini API key. Uses default if None.
            default_api_key: Default API key if api_key is None.
        """
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device_obj = torch.device(self.device)
        self.api_key = api_key or default_api_key
        self.sv_embedder = None
        
    def generate_reference_audio(
        self,
        text: str,
        language_code: str,
        output_path: Path,
        gender: str = "female",
        sampling_rate: int = 8000,
    ) -> bool:
        """
        Generate reference audio using Bhashini TTS API.
        
        Args:
            text: Text to synthesize
            language_code: Bhashini API language code
            output_path: Path to save the audio file
            gender: Gender of voice (default: female)
            sampling_rate: Audio sampling rate (default: 8000)
        
        Returns:
            True if successful, False otherwise
        """
        url = "https://dhruva-api.bhashini.gov.in/services/inference/pipeline"
        
        headers = {
            "Accept": "*/*",
            "User-Agent": "Python-Requests",
            "Authorization": self.api_key,
            "Content-Type": "application/json"
        }
        
        payload = {
            "pipelineTasks": [
                {
                    "taskType": "tts",
                    "config": {
                        "language": {
                            "sourceLanguage": language_code
                        },
                        "serviceId": "Bhashini/IITM/TTS",
                        "gender": gender,
                        "samplingRate": sampling_rate
                    }
                }
            ],
            "inputData": {
                "input": [
                    {
                        "source": text
                    }
                ]
            }
        }
        
        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
            
            if response.status_code != 200:
                print(f"  ❌ API returned status code {response.status_code}")
                print(f"  Response: {response.text[:200]}")
                return False
            
            response_data = response.json()
            
            if "pipelineResponse" not in response_data or not response_data["pipelineResponse"]:
                print(f"  ❌ Invalid API response structure")
                return False
            
            if "audio" not in response_data["pipelineResponse"][0] or not response_data["pipelineResponse"][0]["audio"]:
                print(f"  ❌ No audio in response")
                return False
            
            b64_audio = response_data["pipelineResponse"][0]["audio"][0]["audioContent"]
            audio_bytes = base64.b64decode(b64_audio)
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(audio_bytes)
            
            return True
            
        except Exception as e:
            print(f"  ❌ Error generating reference audio: {e}")
            return False
    
    def generate_all_reference_audios(
        self,
        output_dir: Path = None,
        languages: List[str] = None,
        prompts: Dict[str, str] = None,
        gender: str = "female",
        sampling_rate: int = 8000,
        existing_reference_audios: Optional[Dict[str, Path]] = None,
    ) -> Dict[str, Path]:
        """
        Generate or load reference audio files for all languages.
        
        Args:
            output_dir: Directory to save reference audio files (only used if existing_reference_audios is None)
            languages: List of language codes (default: DEFAULT_LANGUAGES)
            prompts: Dictionary mapping language codes to prompts (default: DEFAULT_PROMPTS)
            gender: Gender of voice (default: female, only used if existing_reference_audios is None)
            sampling_rate: Audio sampling rate (default: 8000, only used if existing_reference_audios is None)
            existing_reference_audios: Optional dictionary mapping language codes to existing reference audio paths.
                                      If provided, these paths will be used directly instead of generating from API.
        
        Returns:
            Dictionary mapping language codes to reference audio paths
        """
        if languages is None:
            languages = DEFAULT_LANGUAGES
        if prompts is None:
            prompts = DEFAULT_PROMPTS
        
        # If existing reference audios are provided, use them directly
        if existing_reference_audios is not None:
            print(f"\n{'='*60}")
            print("Using Existing Reference Audio Files")
            print(f"{'='*60}\n")
            
            reference_audios = {}
            for lang in languages:
                if lang in existing_reference_audios:
                    ref_path = Path(existing_reference_audios[lang])
                    if ref_path.exists():
                        reference_audios[lang] = ref_path
                        print(f"  ✅ Found {lang}: {ref_path}")
                    else:
                        print(f"  ⚠️  Reference audio not found for {lang}: {ref_path}")
                else:
                    print(f"  ⚠️  No reference audio provided for {lang}")
            
            print(f"\n✅ Loaded {len(reference_audios)} reference audio files")
            return reference_audios
        
        # Otherwise, generate from API
        if output_dir is None:
            raise ValueError("output_dir is required when existing_reference_audios is not provided")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        reference_audios = {}
        
        print(f"\n{'='*60}")
        print("Generating Reference Audio Files")
        print(f"{'='*60}\n")
        
        for lang in languages:
            if lang not in prompts:
                print(f"⚠️  Skipping {lang}: No evaluation text available")
                continue
            
            if lang not in BHASHINI_LANG_MAP:
                print(f"⚠️  Skipping {lang}: No Bhashini language mapping")
                continue
            
            text = prompts[lang]
            bhashini_lang = BHASHINI_LANG_MAP[lang]
            output_path = output_dir / f"reference_{lang}.wav"
            
            print(f"Generating reference audio for {lang} ({bhashini_lang})...")
            
            if self.generate_reference_audio(
                text=text,
                language_code=bhashini_lang,
                output_path=output_path,
                gender=gender,
                sampling_rate=sampling_rate,
            ):
                reference_audios[lang] = output_path
                print(f"  ✅ Saved reference audio to {output_path}")
            else:
                print(f"  ⚠️  Failed to generate reference audio for {lang}")
            
            time.sleep(1)  # Rate limiting
        
        print(f"\n✅ Generated {len(reference_audios)} reference audio files")
        return reference_audios
    
    def load_xtts_model(
        self,
        config_path: Path,
        checkpoint_path: Path,
        vocab_path: Path,
    ) -> Xtts:
        """Load XTTS model from checkpoint."""
        config = XttsConfig()
        config.load_json(str(config_path))
        model = Xtts.init_from_config(config)
        model.load_checkpoint(
            config,
            checkpoint_dir=str(checkpoint_path.parent),
            checkpoint_path=str(checkpoint_path),
            vocab_path=str(vocab_path),
            use_deepspeed=False,
        )
        model.to(self.device_obj)
        model.eval()
        return model
    
    def load_audio(self, path: Path, target_sr: int = TARGET_SAMPLE_RATE) -> Tuple[torch.Tensor, int]:
        """Load and resample audio file."""
        wav, sr = torchaudio.load(path)
        if wav.ndim > 1 and wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != target_sr:
            wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
            sr = target_sr
        return wav, sr
    
    def prepare_conditioning(self, model: Xtts, speaker_audio: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare conditioning latents from speaker audio."""
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
            audio_path=str(speaker_audio),
            gpt_cond_len=model.config.gpt_cond_len,
            max_ref_length=model.config.max_ref_len,
            sound_norm_refs=model.config.sound_norm_refs,
        )
        return gpt_cond_latent, speaker_embedding
    
    def synthesize(
        self,
        model: Xtts,
        text: str,
        language: str,
        gpt_latent: torch.Tensor,
        speaker_embedding: torch.Tensor,
        temperature: float = 0.1,
    ) -> Tuple[torch.Tensor, float, float]:
        """
        Synthesize audio and return audio tensor, average latency, and total audio duration.
        
        Returns:
            (audio_tensor, avg_latency_sec, audio_duration_sec)
        """
        chunks = sent_tokenize(text)
        chunk_tensors = []
        latency_budget: List[float] = []
        
        for chunk in tqdm(chunks, desc=f"TTS-{language}", leave=False):
            start = time.perf_counter()
            wav_dict = model.inference(
                text=chunk,
                language=language,
                gpt_cond_latent=gpt_latent,
                speaker_embedding=speaker_embedding,
                temperature=temperature,
                length_penalty=1.0,
                repetition_penalty=10.0,
                top_k=10,
                top_p=0.3,
            )
            latency_budget.append(time.perf_counter() - start)
            chunk_tensors.append(torch.tensor(wav_dict["wav"], device=self.device_obj))
        
        audio = torch.cat(chunk_tensors, dim=0).unsqueeze(0).cpu()
        avg_latency = float(np.mean(latency_budget))
        audio_duration = audio.shape[-1] / TARGET_SAMPLE_RATE
        return audio, avg_latency, audio_duration
    
    def init_speaker_verifier(self):
        """Initialize speaker verification model for similarity computation."""
        if self.sv_embedder is not None:
            return self.sv_embedder
        
        if hasattr(torchaudio.pipelines, "SUPERB_SV"):
            pipeline = torchaudio.pipelines.SUPERB_SV
            model = pipeline.get_model().to(self.device_obj)
            model.eval()
            
            def embed_superb(wav: torch.Tensor) -> torch.Tensor:
                if wav.shape[0] > 1:
                    wav_proc = wav.mean(dim=0, keepdim=True)
                else:
                    wav_proc = wav
                if wav_proc.shape[-1] < pipeline.sample_rate:
                    wav_proc = torch.nn.functional.pad(
                        wav_proc, (0, pipeline.sample_rate - wav_proc.shape[-1])
                    )
                else:
                    wav_proc = wav_proc[..., : pipeline.sample_rate]
                with torch.inference_mode():
                    return model(wav_proc.to(self.device_obj))
            
            self.sv_embedder = embed_superb
            return self.sv_embedder
        
        bundle = torchaudio.pipelines.WAV2VEC2_BASE
        model = bundle.get_model().to(self.device_obj)
        model.eval()
        resampler = torchaudio.transforms.Resample(
            orig_freq=TARGET_SAMPLE_RATE, new_freq=bundle.sample_rate
        )
        
        def embed_wav2vec(wav: torch.Tensor) -> torch.Tensor:
            if wav.shape[0] > 1:
                wav_proc = wav.mean(dim=0, keepdim=True)
            else:
                wav_proc = wav
            wav_proc = resampler(wav_proc.cpu()).to(self.device_obj)
            with torch.inference_mode():
                features, _ = model.extract_features(wav_proc)
                embedding = features[-1].mean(dim=1)
                embedding = torch.nn.functional.normalize(embedding, dim=-1)
                return embedding
        
        self.sv_embedder = embed_wav2vec
        return self.sv_embedder
    
    def compute_speaker_similarity(
        self,
        ref_wav: torch.Tensor,
        gen_wav: torch.Tensor,
    ) -> float:
        """Compute cosine similarity between reference and generated audio embeddings."""
        if self.sv_embedder is None:
            self.init_speaker_verifier()
        
        with torch.inference_mode():
            ref_emb = self.sv_embedder(ref_wav)
            gen_emb = self.sv_embedder(gen_wav)
        cos = torch.nn.functional.cosine_similarity(ref_emb, gen_emb)
        return float(cos.mean().cpu())
    
    def compute_mos(self, wav: torch.Tensor) -> float:
        """
        Compute automatic MOS using wav2vec2-based approach.
        This is a simplified proxy for MOS - actual MOS requires human evaluation.
        """
        try:
            bundle = torchaudio.pipelines.WAV2VEC2_BASE
            model = bundle.get_model().to(self.device_obj)
            model.eval()
            resampler = torchaudio.transforms.Resample(
                orig_freq=TARGET_SAMPLE_RATE, new_freq=bundle.sample_rate
            )
            
            if wav.shape[0] > 1:
                wav_proc = wav.mean(dim=0, keepdim=True)
            else:
                wav_proc = wav
            
            wav_proc = resampler(wav_proc.cpu()).to(self.device_obj)
            
            with torch.inference_mode():
                features, _ = model.extract_features(wav_proc)
                last_layer = features[-1]
                energy = torch.mean(torch.abs(last_layer))
                variance = torch.var(last_layer)
                mos_proxy = 1.0 + 4.0 * torch.sigmoid(energy * 0.1 + variance * 0.01)
                return float(mos_proxy.cpu().item())
        except Exception as e:
            print(f"[WARN] MOS calculation failed: {e}")
            return 0.0
    
    def compute_pesq(
        self,
        ref_wav: torch.Tensor,
        deg_wav: torch.Tensor,
        sample_rate: int = TARGET_SAMPLE_RATE
    ) -> Optional[float]:
        """Compute PESQ score between reference and degraded audio."""
        if not PESQ_AVAILABLE:
            return None
        
        try:
            if ref_wav.ndim == 1:
                ref_wav = ref_wav.unsqueeze(0)
            if deg_wav.ndim == 1:
                deg_wav = deg_wav.unsqueeze(0)
            
            min_len = min(ref_wav.shape[-1], deg_wav.shape[-1])
            ref_wav = ref_wav[..., :min_len]
            deg_wav = deg_wav[..., :min_len]
            
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                ref_wav = resampler(ref_wav)
                deg_wav = resampler(deg_wav)
                sample_rate = 16000
            
            pesq_metric = PerceptualEvaluationSpeechQuality(fs=sample_rate, mode="wb")
            pesq_score = pesq_metric(ref_wav, deg_wav)
            
            if isinstance(pesq_score, torch.Tensor):
                return float(pesq_score.item())
            return float(pesq_score)
        except Exception as e:
            print(f"[WARN] PESQ calculation failed: {e}")
            return None
    
    def compute_mcd(
        self,
        ref_wav: torch.Tensor,
        gen_wav: torch.Tensor,
        sample_rate: int = TARGET_SAMPLE_RATE
    ) -> Optional[float]:
        """Compute Mel Cepstral Distortion (MCD) between reference and generated audio."""
        if not LIBROSA_AVAILABLE:
            return None
        
        try:
            ref_np = ref_wav.squeeze().cpu().numpy()
            gen_np = gen_wav.squeeze().cpu().numpy()
            
            min_len = min(len(ref_np), len(gen_np))
            ref_np = ref_np[:min_len]
            gen_np = gen_np[:min_len]
            
            ref_mfcc = librosa.feature.mfcc(y=ref_np, sr=sample_rate, n_mfcc=13)
            gen_mfcc = librosa.feature.mfcc(y=gen_np, sr=sample_rate, n_mfcc=13)
            
            min_frames = min(ref_mfcc.shape[1], gen_mfcc.shape[1])
            ref_mfcc = ref_mfcc[:, :min_frames]
            gen_mfcc = gen_mfcc[:, :min_frames]
            
            mcd = np.mean(np.sqrt(np.sum((ref_mfcc[1:] - gen_mfcc[1:]) ** 2, axis=0)))
            return float(mcd)
        except Exception as e:
            print(f"[WARN] MCD calculation failed: {e}")
            return None
    
    def get_model_size(self, checkpoint_path: Path) -> Dict[str, float]:
        """Calculate model size in bytes, MB, and GB."""
        if not checkpoint_path.exists():
            return {"bytes": 0, "mb": 0.0, "gb": 0.0}
        
        size_bytes = checkpoint_path.stat().st_size
        return {
            "bytes": size_bytes,
            "mb": size_bytes / (1024 * 1024),
            "gb": size_bytes / (1024 * 1024 * 1024),
        }
    
    def find_checkpoints(self, checkpoint_dir: Path) -> List[Dict[str, Path]]:
        """Find all checkpoints in a directory structure."""
        checkpoint_dir = Path(checkpoint_dir)
        checkpoints = []
        
        for root, dirs, files in os.walk(checkpoint_dir):
            root_path = Path(root)
            config_path = root_path / "config.json"
            
            if config_path.exists():
                checkpoint_files = list(root_path.glob("*.pth"))
                checkpoint_files = [f for f in checkpoint_files if "checkpoint" in f.name or "best_model" in f.name]
                
                vocab_path = root_path / "vocab.json"
                if not vocab_path.exists():
                    vocab_path = root_path / "XTTS_v2.0_original_model_files" / "vocab.json"
                
                if not vocab_path.exists():
                    current = root_path.parent
                    max_depth = 5
                    depth = 0
                    while current != current.parent and depth < max_depth:
                        vocab_path = current / "vocab.json"
                        if vocab_path.exists():
                            break
                        vocab_path = current / "XTTS_v2.0_original_model_files" / "vocab.json"
                        if vocab_path.exists():
                            break
                        current = current.parent
                        depth += 1
                
                for checkpoint_file in checkpoint_files:
                    if vocab_path.exists():
                        checkpoints.append({
                            "checkpoint": checkpoint_file,
                            "config": config_path,
                            "vocab": vocab_path,
                            "name": f"{root_path.name}_{checkpoint_file.stem}",
                        })
        
        return checkpoints
    
    def evaluate_checkpoint(
        self,
        checkpoint_path: Path,
        config_path: Path,
        vocab_path: Path,
        speaker_audio: Path,
        reference_audio: Optional[Path],
        output_dir: Path,
        languages: List[str],
        prompts: Dict[str, str] = None,
        max_prompts: int = None,
        temperature: float = 0.1,
    ) -> Dict:
        """
        Evaluate a single checkpoint and return metrics.
        
        Args:
            checkpoint_path: Path to checkpoint .pth file
            config_path: Path to config.json
            vocab_path: Path to vocab.json
            speaker_audio: Reference speaker audio for zero-shot cloning
            reference_audio: Optional reference audio for PESQ/MCD metrics
            output_dir: Directory to save outputs
            languages: List of language codes to evaluate
            prompts: Dictionary mapping language codes to prompts (default: DEFAULT_PROMPTS)
            max_prompts: Maximum number of languages to evaluate (None = all)
            temperature: Temperature for TTS generation
        
        Returns:
            Dictionary containing all metrics
        """
        if prompts is None:
            prompts = DEFAULT_PROMPTS
        
        # Convert all path parameters to Path objects
        checkpoint_path = Path(checkpoint_path)
        config_path = Path(config_path)
        vocab_path = Path(vocab_path)
        speaker_audio = Path(speaker_audio)
        output_dir = Path(output_dir)
        if reference_audio is not None:
            reference_audio = Path(reference_audio)
        
        print(f"\n{'='*60}")
        print(f"Evaluating checkpoint: {checkpoint_path.name}")
        print(f"{'='*60}")
        
        checkpoint_output_dir = output_dir / checkpoint_path.stem
        checkpoint_output_dir.mkdir(parents=True, exist_ok=True)
        
        model = self.load_xtts_model(config_path, checkpoint_path, vocab_path)
        gpt_latent, speaker_embedding = self.prepare_conditioning(model, speaker_audio)
        ref_audio, ref_sr = self.load_audio(speaker_audio, target_sr=24000)
        
        ref_audio_for_metrics = None
        if reference_audio and reference_audio.exists():
            ref_audio_for_metrics, _ = self.load_audio(reference_audio, target_sr=24000)
        
        self.init_speaker_verifier()
        model_size = self.get_model_size(checkpoint_path)
        
        metrics = {
            "checkpoint": str(checkpoint_path),
            "checkpoint_name": checkpoint_path.stem,
            "config": str(config_path),
            "vocab": str(vocab_path),
            "speaker_audio": str(speaker_audio),
            "reference_audio": str(reference_audio) if reference_audio else None,
            "model_size_mb": model_size["mb"],
            "model_size_gb": model_size["gb"],
            "languages": {},
        }
        
        languages_to_eval = languages[:max_prompts] if max_prompts else languages
        
        for lang in languages_to_eval:
            prompt = prompts.get(lang.lower())
            if not prompt:
                print(f"[WARN] No default prompt for {lang}, skipping.")
                continue
            
            print(f"\n=== Evaluating {lang} ===")
            audio, latency, audio_duration = self.synthesize(
                model=model,
                text=prompt,
                language=lang,
                gpt_latent=gpt_latent,
                speaker_embedding=speaker_embedding,
                temperature=temperature,
            )
            
            output_wav = checkpoint_output_dir / f"tts_{lang}.wav"
            torchaudio.save(str(output_wav), audio, 24000)
            
            gen_similarity = self.compute_speaker_similarity(ref_audio, audio)
            mos_score = self.compute_mos(audio)
            rtf = latency / audio_duration if audio_duration > 0 else 0.0
            
            pesq_score = None
            mcd_score = None
            if ref_audio_for_metrics is not None:
                pesq_score = self.compute_pesq(ref_audio_for_metrics, audio, TARGET_SAMPLE_RATE)
                mcd_score = self.compute_mcd(ref_audio_for_metrics, audio, TARGET_SAMPLE_RATE)
            
            metrics["languages"][lang] = {
                "latency_sec": latency,
                "audio_duration_sec": audio_duration,
                "rtf": rtf,
                "audio_path": str(output_wav),
                "speaker_similarity": gen_similarity,
                "mos": mos_score,
                "pesq": pesq_score,
                "mcd": mcd_score,
            }
            
            print(f"  Latency: {latency:.3f}s")
            print(f"  RTF: {rtf:.3f}")
            print(f"  Speaker Similarity: {gen_similarity:.3f}")
            print(f"  MOS: {mos_score:.3f}")
            if pesq_score is not None:
                print(f"  PESQ: {pesq_score:.3f}")
            if mcd_score is not None:
                print(f"  MCD: {mcd_score:.3f}")
        
        return metrics
    
    def evaluate_checkpoints(
        self,
        checkpoint_dir: Path,
        speaker_audio: Path,
        reference_audios: Optional[Dict[str, Path]] = None,
        reference_audio_dir: Optional[Path] = None,
        output_dir: Path = None,
        languages: List[str] = None,
        prompts: Dict[str, str] = None,
        max_prompts: int = None,
        temperature: float = 0.1,
        per_language_reference: bool = True,
    ) -> Dict:
        """
        Evaluate multiple checkpoints in a directory.
        
        Args:
            checkpoint_dir: Directory containing checkpoints
            speaker_audio: Reference speaker audio for zero-shot cloning
            reference_audios: Dictionary mapping language codes to reference audio paths.
                             If None and reference_audio_dir is provided, will load from directory.
            reference_audio_dir: Optional directory containing reference audio files.
                                Files should be named as "reference_{lang}.wav" (e.g., "reference_hi.wav").
                                If provided and reference_audios is None, will automatically load files from this directory.
            output_dir: Directory to save outputs (default: checkpoint_dir / "evaluation_results")
            languages: List of language codes to evaluate (default: DEFAULT_LANGUAGES)
            prompts: Dictionary mapping language codes to prompts (default: DEFAULT_PROMPTS)
            max_prompts: Maximum number of languages to evaluate (None = all)
            temperature: Temperature for TTS generation
            per_language_reference: If True, use language-specific reference audio for each language
        
        Returns:
            Dictionary containing all metrics for all checkpoints
        """
        # Convert all path parameters to Path objects
        checkpoint_dir = Path(checkpoint_dir)
        speaker_audio = Path(speaker_audio)
        
        if languages is None:
            languages = DEFAULT_LANGUAGES
        if prompts is None:
            prompts = DEFAULT_PROMPTS
        if output_dir is None:
            output_dir = checkpoint_dir / "evaluation_results"
        else:
            output_dir = Path(output_dir)
        
        # Load reference audios from directory if reference_audios is None but reference_audio_dir is provided
        if reference_audios is None and reference_audio_dir is not None:
            reference_audio_dir = Path(reference_audio_dir)
            if reference_audio_dir.exists() and reference_audio_dir.is_dir():
                print(f"\n{'='*60}")
                print("Loading Reference Audio Files from Directory")
                print(f"{'='*60}\n")
                reference_audios = {}
                for lang in languages:
                    ref_path = reference_audio_dir / f"reference_{lang}.wav"
                    if ref_path.exists():
                        reference_audios[lang] = ref_path
                        print(f"  ✅ Found {lang}: {ref_path}")
                    else:
                        print(f"  ⚠️  Reference audio not found for {lang}: {ref_path}")
                print(f"\n✅ Loaded {len(reference_audios)} reference audio files from {reference_audio_dir}")
            else:
                print(f"[WARN] Reference audio directory does not exist: {reference_audio_dir}")
                reference_audios = None
        
        checkpoints = self.find_checkpoints(checkpoint_dir)
        if not checkpoints:
            print(f"[ERROR] No checkpoints found in {checkpoint_dir}")
            return {}
        
        print(f"Found {len(checkpoints)} checkpoint(s) to evaluate")
        
        all_metrics = {
            "evaluation_summary": {
                "total_checkpoints": len(checkpoints),
                "speaker_audio": str(speaker_audio),
                "languages": languages[:max_prompts] if max_prompts else languages,
            },
            "checkpoints": [],
        }
        
        for ckpt_info in tqdm(checkpoints, desc="Evaluating checkpoints"):
            try:
                if per_language_reference and reference_audios:
                    # Evaluate each language separately with its reference audio
                    checkpoint_metrics = {
                        "checkpoint": str(ckpt_info["checkpoint"]),
                        "checkpoint_name": ckpt_info["name"],
                        "config": str(ckpt_info["config"]),
                        "vocab": str(ckpt_info["vocab"]),
                        "speaker_audio": str(speaker_audio),
                        "languages": {},
                    }
                    
                    for lang in languages[:max_prompts] if max_prompts else languages:
                        lang_ref_audio = reference_audios.get(lang)
                        lang_output_dir = output_dir / ckpt_info["name"] / lang
                        
                        metrics = self.evaluate_checkpoint(
                            checkpoint_path=ckpt_info["checkpoint"],
                            config_path=ckpt_info["config"],
                            vocab_path=ckpt_info["vocab"],
                            speaker_audio=speaker_audio,
                            reference_audio=lang_ref_audio,
                            output_dir=lang_output_dir,
                            languages=[lang],
                            prompts=prompts,
                            max_prompts=None,
                            temperature=temperature,
                        )
                        checkpoint_metrics["languages"][lang] = metrics.get("languages", {}).get(lang, {})
                    
                    all_metrics["checkpoints"].append(checkpoint_metrics)
                else:
                    # Evaluate all languages at once with single reference audio (or none)
                    ref_audio = None
                    if reference_audios and len(reference_audios) == 1:
                        ref_audio = list(reference_audios.values())[0]
                    elif reference_audios:
                        # Use first language's reference audio
                        first_lang = languages[0] if languages else list(reference_audios.keys())[0]
                        ref_audio = reference_audios.get(first_lang)
                    
                    metrics = self.evaluate_checkpoint(
                        checkpoint_path=ckpt_info["checkpoint"],
                        config_path=ckpt_info["config"],
                        vocab_path=ckpt_info["vocab"],
                        speaker_audio=speaker_audio,
                        reference_audio=ref_audio,
                        output_dir=output_dir,
                        languages=languages,
                        prompts=prompts,
                        max_prompts=max_prompts,
                        temperature=temperature,
                    )
                    all_metrics["checkpoints"].append(metrics)
            except Exception as e:
                print(f"\n[ERROR] Failed to evaluate {ckpt_info['checkpoint']}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Save combined metrics
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, ensure_ascii=False, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Evaluation complete. Report saved to {metrics_path}")
        print(f"Evaluated {len(all_metrics['checkpoints'])} checkpoint(s)")
        print(f"{'='*60}")
        
        return all_metrics

if __name__ == "__main__":
    metrics = AkaalTTSMetrics(device="cuda:0", api_key="5Yknj5sVdr1hA4V7JoKrzISnVfnXl5iUCnbZ9NZTbUrcudoAOzq0fc6VzQF19iB9")
    metrics.evaluate_checkpoints(
        checkpoint_dir="/home/ubuntu/sneh/XTTSv2-Finetuning-for-New-Languages/checkpoints/GPT_XTTS_FT-November-20-2025_05+30PM-8e59ec3",
        speaker_audio="/home/ubuntu/sneh/output.wav",
        output_dir="/home/ubuntu/sneh/voicecraft-x/evalutaion",
        reference_audio_dir="/home/ubuntu/sneh/evaluation_results/reference_audio",
        languages=["hi", "en", "guj"],
        per_language_reference=True
    )
