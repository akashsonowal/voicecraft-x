#!/usr/bin/env python3
"""
Refactored EDA script for medical TTS dataset with Indic language support.
Fixes: SNR calculation, speech rate for Indic languages, text normalization, pitch estimation.
"""

import os
import sys
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from scipy.stats import norm
from tqdm import tqdm
from multiprocessing import Pool
import soundfile as sf
import json
from typing import List, Dict, Tuple, Optional
import unicodedata
import gc


# Configuration
BATCH_SIZE = 100
NUM_PROC = 8
DATA_ROOT = "/home/ubuntu/sneh/data"
OUTPUT_DIR = "/home/ubuntu/sneh/analysis_results"

HEALTH_KEYWORDS = ["health", "heal"]
LOCAL_CONVERSATION_KEYWORDS = ["local", "conversation", "loca"]

# Language-specific parameters
LANGUAGE_NORMAL_WPM = {
    'hindi': 150,
    'bengali': 140,
    'bhojpuri': 145,
    'marathi': 150,
    'chhattisgarhi': 140,
    'magahi': 145,
    'mathili': 145,
    'tamil': 130,
    'kannada': 135,
    'telugu': 140,
    'gujarati': 150,
    'english': 150
}

PITCH_RANGES = {
    'male': {'fmin': 50, 'fmax': 180},
    'female': {'fmin': 140, 'fmax': 300},
    'unknown': {'fmin': 50, 'fmax': 300}
}

MEDICAL_QUALITY_THRESHOLDS = {
    'snr_min': 15,
    'snr_good': 25,
    'speech_rate_tolerance': 0.15,
    'audio_duration_min': 1.0,
    'audio_duration_max': 180.0
}


def is_target_domain(domain: str) -> bool:
    """Check if domain matches HEALTH or LOCAL CONVERSATION criteria."""
    domain_lower = domain.lower().strip()
    for keyword in HEALTH_KEYWORDS:
        if keyword in domain_lower:
            return True
    for keyword in LOCAL_CONVERSATION_KEYWORDS:
        if keyword in domain_lower:
            return True
    return False


def normalize_text(text: str) -> str:
    """Normalize Unicode and handle script-specific issues."""
    text = unicodedata.normalize('NFC', text)
    text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C')
    text = text.strip()
    return text


def formatter(root_path: str, meta_file: str) -> List[Dict]:
    """Load metadata.csv and return items matching target domains."""
    txt_file = os.path.join(root_path, meta_file)
    items = []
    
    if not os.path.exists(txt_file):
        return items
    
    with open(txt_file, "r", encoding="utf-8") as ttf:
        for line in ttf:
            line = line.strip()
            if not line:
                continue
            cols = line.split("|")
            if len(cols) < 3:
                continue
            
            file_id = cols[0].strip()
            text = normalize_text(cols[1].strip())
            domain = cols[2].strip()
            language = cols[3].strip().lower() if len(cols) > 3 else ""
            
            if not is_target_domain(domain):
                continue
            
            wav_file = os.path.join(root_path, "wavs", file_id + ".wav")
            if not os.path.exists(wav_file):
                wav_file = os.path.join(root_path, "wav", file_id + ".wav")
            
            if not os.path.exists(wav_file):
                continue
            
            items.append({
                "text": text,
                "audio_file": wav_file,
                "speaker_name": f"myspeaker{language}",
                "root_path": root_path,
                "language": language,
                "domain": domain,
                "file_id": file_id
            })
    
    return items


def compute_snr_spectral(wav_path: str) -> float:
    """
    Compute SNR using spectral-based estimation (robust for speech).
    Estimates noise from low-energy frames.
    """
    try:
        audio, sr = sf.read(wav_path)
        
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        # Compute STFT
        D = librosa.stft(audio)
        S = np.abs(D) ** 2
        
        # Frame energy
        frame_energy = np.mean(S, axis=0)
        
        # Noise = frames in quietest 20th percentile
        noise_threshold = np.percentile(frame_energy, 20)
        noise_frames = frame_energy < noise_threshold
        
        if np.sum(noise_frames) == 0:
            return np.nan
        
        noise_power = np.mean(frame_energy[noise_frames])
        signal_power = np.mean(frame_energy[~noise_frames])
        
        if noise_power == 0 or signal_power == 0:
            return np.nan
        
        snr = 10 * np.log10(signal_power / noise_power)
        return float(snr)
    except Exception as e:
        print(f"SNR error for {wav_path}: {e}")
        return np.nan


def compute_pitch(audio: np.ndarray, sr: int) -> Dict:
    """Compute pitch statistics with downsampling and language-aware ranges."""
    try:
        # Downsample to 16kHz (optimal for PYIN)
        if sr > 16000:
            audio_ds = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr_ds = 16000
        else:
            audio_ds = audio
            sr_ds = sr
        
        # Use default pitch range
        pitch_range = PITCH_RANGES['unknown']
        
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio_ds,
            fmin=pitch_range['fmin'],
            fmax=pitch_range['fmax']
        )
        
        f0_voiced = f0[~np.isnan(f0)]
        
        if len(f0_voiced) == 0:
            return {
                "mean_pitch": np.nan,
                "median_pitch": np.nan,
                "pitch_std": np.nan,
                "pitch_range": np.nan,
                "voiced_ratio": 0.0
            }
        
        return {
            "mean_pitch": float(np.mean(f0_voiced)),
            "median_pitch": float(np.median(f0_voiced)),
            "pitch_std": float(np.std(f0_voiced)),
            "pitch_range": float(np.max(f0_voiced) - np.min(f0_voiced)),
            "voiced_ratio": float(np.sum(~np.isnan(f0)) / len(f0))
        }
    except Exception as e:
        print(f"Pitch error: {e}")
        return {
            "mean_pitch": np.nan,
            "median_pitch": np.nan,
            "pitch_std": np.nan,
            "pitch_range": np.nan,
            "voiced_ratio": 0.0
        }


def compute_speech_rate(text: str, audio_duration: float, language: str) -> Dict:
    """
    Compute speech rate metrics with language-aware tokenization.
    Uses character-based rate (language-agnostic) + word-based rate.
    """
    text_clean = normalize_text(text)
    
    num_chars = len(text_clean)
    chars_per_second = num_chars / audio_duration if audio_duration > 0 else 0.0
    
    # Word-based metrics with fallback
    try:
        from indic_nlp.tokenize import indic_tokenize
        indic_langs = ['hindi', 'marathi', 'bhojpuri', 'bengali', 'tamil', 'kannada', 
                      'telugu', 'gujarati', 'chhattisgarhi', 'magahi', 'mathili']
        
        if language.lower() in indic_langs:
            words = indic_tokenize.trivial_tokenize(text_clean, lang=language[:2].lower())
        else:
            words = text_clean.split()
    except:
        words = text_clean.split()
    
    num_words = len(words)
    wpm = (num_words / audio_duration) * 60.0 if audio_duration > 0 else 0.0
    
    normal_wpm = LANGUAGE_NORMAL_WPM.get(language.lower(), 150)
    wpm_ratio = wpm / normal_wpm if normal_wpm > 0 else 0.0
    
    is_slow = wpm < (normal_wpm * 0.8)
    is_fast = wpm > (normal_wpm * 1.2)
    
    return {
        "num_chars": int(num_chars),
        "num_words": int(num_words),
        "chars_per_second": float(chars_per_second),
        "words_per_minute": float(wpm),
        "normal_wpm": float(normal_wpm),
        "wpm_ratio": float(wpm_ratio),
        "is_slow": bool(is_slow),
        "is_fast": bool(is_fast),
        "duration_sec": float(audio_duration)
    }


def detect_clipping(audio: np.ndarray, threshold: float = 0.99) -> Dict:
    """Detect audio clipping (saturation)."""
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio_norm = audio / max_val
    else:
        audio_norm = audio
    
    clipped_samples = np.sum(np.abs(audio_norm) > threshold)
    clipped_ratio = clipped_samples / len(audio) if len(audio) > 0 else 0
    
    return {
        'clipped_ratio': float(clipped_ratio),
        'is_clipped': clipped_ratio > 0.01,
        'clipping_severity': 'severe' if clipped_ratio > 0.05 else 'mild' if clipped_ratio > 0.01 else 'none'
    }


def flag_low_quality(item_stats: Dict) -> Dict:
    """Flag items unsuitable for medical TTS."""
    flags = []
    
    snr = item_stats.get('snr')
    if np.isnan(snr):
        flags.append('snr_calculation_failed')
    elif snr < MEDICAL_QUALITY_THRESHOLDS['snr_min']:
        flags.append(f'low_snr_{snr:.1f}db')
    elif snr < MEDICAL_QUALITY_THRESHOLDS['snr_good']:
        flags.append(f'marginal_snr_{snr:.1f}db')
    
    if item_stats.get('is_clipped', False):
        severity = item_stats.get('clipping_severity', 'unknown')
        flags.append(f'clipping_{severity}')
    
    duration = item_stats.get('duration_sec', 0)
    if duration < MEDICAL_QUALITY_THRESHOLDS['audio_duration_min']:
        flags.append('too_short')
    elif duration > MEDICAL_QUALITY_THRESHOLDS['audio_duration_max']:
        flags.append('too_long')
    
    wpm_ratio = item_stats.get('wpm_ratio', 1.0)
    if abs(wpm_ratio - 1.0) > MEDICAL_QUALITY_THRESHOLDS['speech_rate_tolerance']:
        flags.append(f'abnormal_speech_rate_{wpm_ratio:.2f}x')
    
    return {
        'quality_flags': flags,
        'is_usable_for_medical': len(flags) == 0,
        'confidence': 'high' if len(flags) == 0 else 'low'
    }


def load_item(item: Dict) -> Tuple:
    """Load audio file and extract all metrics."""
    try:
        text = item["text"]
        file_name = item["audio_file"]
        language = item["language"]
        
        audio, sr = librosa.load(file_name, sr=None)
        audio_len = len(audio) / sr
        
        pitch_stats = compute_pitch(audio, sr)
        speech_rate_stats = compute_speech_rate(text, audio_len, language)
        clipping_stats = detect_clipping(audio)
        
        return (
            file_name, text, audio_len, audio, sr,
            item["domain"], language,
            pitch_stats, speech_rate_stats, clipping_stats
        )
    except Exception as e:
        print(f"Error loading {item.get('audio_file', 'unknown')}: {e}")
        return None


def process_batch(items: List[Dict], batch_num: int, total_batches: int) -> Dict:
    """Process a batch of items and return statistics."""
    print(f"\nProcessing batch {batch_num}/{total_batches} ({len(items)} items)...")
    
    if NUM_PROC == 1:
        data = []
        for item in tqdm(items, desc=f"Loading batch {batch_num}"):
            result = load_item(item)
            if result is not None:
                data.append(result)
    else:
        with Pool(NUM_PROC) as p:
            results = list(tqdm(p.imap(load_item, items), total=len(items), desc=f"Loading batch {batch_num}"))
            data = [r for r in results if r is not None]
    
    if not data:
        return {}
    
    print(f"Computing SNR for batch {batch_num}...")
    snrs = []
    quality_assessments = []
    
    for item in tqdm(data, desc=f"SNR batch {batch_num}"):
        file_name = item[0]
        snr = compute_snr_spectral(file_name)
        snrs.append(snr)
        
        item_stats = {
            'snr': snr,
            'is_clipped': item[9].get('is_clipped', False),
            'clipping_severity': item[9].get('clipping_severity', 'none'),
            'duration_sec': item[2],
            'wpm_ratio': item[8].get('wpm_ratio', 1.0)
        }
        quality_assessment = flag_low_quality(item_stats)
        quality_assessments.append(quality_assessment)
    
    audio_lens = [item[2] for item in data]
    text_lens = [len(item[1]) for item in data]
    domains = [item[5] for item in data]
    languages = [item[6] for item in data]
    texts = [item[1] for item in data]
    pitch_stats_list = [item[7] for item in data]
    speech_rate_stats_list = [item[8] for item in data]
    clipping_stats_list = [item[9] for item in data]
    
    w_count = Counter()
    for text in texts:
        for word in text.lower().strip().split():
            w_count[word] += 1
    
    text_vs_durs = {}
    for text_len, audio_len in zip(text_lens, audio_lens):
        if text_len not in text_vs_durs:
            text_vs_durs[text_len] = []
        text_vs_durs[text_len].append(audio_len)
    
    valid_snrs = [s for s in snrs if not np.isnan(s) and not np.isinf(s)]
    
    mean_pitches = [ps["mean_pitch"] for ps in pitch_stats_list if not np.isnan(ps["mean_pitch"])]
    median_pitches = [ps["median_pitch"] for ps in pitch_stats_list if not np.isnan(ps["median_pitch"])]
    pitch_stds = [ps["pitch_std"] for ps in pitch_stats_list if not np.isnan(ps["pitch_std"])]
    
    wpms = [srs["words_per_minute"] for srs in speech_rate_stats_list]
    cpss = [srs["chars_per_second"] for srs in speech_rate_stats_list]
    wpm_ratios = [srs["wpm_ratio"] for srs in speech_rate_stats_list]
    slow_count = sum(1 for srs in speech_rate_stats_list if srs["is_slow"])
    fast_count = sum(1 for srs in speech_rate_stats_list if srs["is_fast"])
    
    clipped_count = sum(1 for cs in clipping_stats_list if cs["is_clipped"])
    usable_count = sum(1 for qa in quality_assessments if qa["is_usable_for_medical"])
    
    batch_stats = {
        "num_items": len(data),
        "text_lens": text_lens,
        "audio_lens": audio_lens,
        "snrs": valid_snrs,
        "domains": domains,
        "languages": languages,
        "word_counts": w_count,
        "text_vs_durs": text_vs_durs,
        "file_names": [item[0] for item in data],
        "mean_pitches": mean_pitches,
        "median_pitches": median_pitches,
        "pitch_stds": pitch_stds,
        "pitch_stats": pitch_stats_list,
        "speech_rate_stats": speech_rate_stats_list,
        "wpms": wpms,
        "cpss": cpss,
        "wpm_ratios": wpm_ratios,
        "slow_count": slow_count,
        "fast_count": fast_count,
        "clipped_count": clipped_count,
        "usable_count": usable_count,
        "quality_assessments": quality_assessments
    }
    
    del data
    gc.collect()
    return batch_stats


def aggregate_statistics(all_batch_stats: List[Dict]) -> Dict:
    """Aggregate statistics from all batches."""
    if not all_batch_stats:
        return {}
    
    total_items = sum(stats["num_items"] for stats in all_batch_stats)
    
    all_text_lens = []
    all_audio_lens = []
    all_snrs = []
    all_domains = []
    all_languages = []
    all_pitch_stats = []
    all_speech_rate_stats = []
    all_wpms = []
    all_cpss = []
    all_quality_assessments = []
    total_slow_count = 0
    total_fast_count = 0
    total_clipped_count = 0
    total_usable_count = 0
    
    combined_word_counts = Counter()
    
    for stats in all_batch_stats:
        all_text_lens.extend(stats["text_lens"])
        all_audio_lens.extend(stats["audio_lens"])
        all_snrs.extend(stats["snrs"])
        all_domains.extend(stats["domains"])
        all_languages.extend(stats["languages"])
        all_pitch_stats.extend(stats.get("pitch_stats", []))
        all_speech_rate_stats.extend(stats.get("speech_rate_stats", []))
        all_wpms.extend(stats.get("wpms", []))
        all_cpss.extend(stats.get("cpss", []))
        all_quality_assessments.extend(stats.get("quality_assessments", []))
        total_slow_count += stats.get("slow_count", 0)
        total_fast_count += stats.get("fast_count", 0)
        total_clipped_count += stats.get("clipped_count", 0)
        total_usable_count += stats.get("usable_count", 0)
        combined_word_counts.update(stats["word_counts"])
    
    combined_text_vs_durs = {}
    for stats in all_batch_stats:
        for text_len, durs in stats["text_vs_durs"].items():
            if text_len not in combined_text_vs_durs:
                combined_text_vs_durs[text_len] = []
            combined_text_vs_durs[text_len].extend(durs)
    
    text_len_counter = Counter(all_text_lens)
    domain_counter = Counter(all_domains)
    language_counter = Counter(all_languages)
    
    avg_snr = np.mean(all_snrs) if all_snrs else np.nan
    median_snr = np.median(all_snrs) if all_snrs else np.nan
    std_snr = np.std(all_snrs) if all_snrs else np.nan
    
    valid_mean_pitches = [ps["mean_pitch"] for ps in all_pitch_stats if not np.isnan(ps.get("mean_pitch", np.nan))]
    valid_median_pitches = [ps["median_pitch"] for ps in all_pitch_stats if not np.isnan(ps.get("median_pitch", np.nan))]
    
    avg_wpm = np.mean(all_wpms) if all_wpms else np.nan
    median_wpm = np.median(all_wpms) if all_wpms else np.nan
    std_wpm = np.std(all_wpms) if all_wpms else np.nan
    avg_cps = np.mean(all_cpss) if all_cpss else np.nan
    
    language_wise_stats = {}
    for lang in set(all_languages):
        lang_indices = [i for i, l in enumerate(all_languages) if l == lang]
        lang_pitch_stats = [all_pitch_stats[i] for i in lang_indices if i < len(all_pitch_stats)]
        lang_speech_stats = [all_speech_rate_stats[i] for i in lang_indices if i < len(all_speech_rate_stats)]
        lang_wpms = [all_wpms[i] for i in lang_indices if i < len(all_wpms)]
        lang_cpss = [all_cpss[i] for i in lang_indices if i < len(all_cpss)]
        lang_snrs = [all_snrs[i] for i in lang_indices if i < len(all_snrs)]
        
        lang_mean_pitches = [ps["mean_pitch"] for ps in lang_pitch_stats if not np.isnan(ps.get("mean_pitch", np.nan))]
        lang_median_pitches = [ps["median_pitch"] for ps in lang_pitch_stats if not np.isnan(ps.get("median_pitch", np.nan))]
        lang_slow_count = sum(1 for srs in lang_speech_stats if srs.get("is_slow", False))
        lang_fast_count = sum(1 for srs in lang_speech_stats if srs.get("is_fast", False))
        
        language_wise_stats[lang] = {
            "count": len(lang_indices),
            "snr": {
                "mean": float(np.mean(lang_snrs)) if lang_snrs else np.nan,
                "median": float(np.median(lang_snrs)) if lang_snrs else np.nan,
                "std": float(np.std(lang_snrs)) if lang_snrs else np.nan,
                "min": float(np.min(lang_snrs)) if lang_snrs else np.nan,
                "max": float(np.max(lang_snrs)) if lang_snrs else np.nan
            },
            "pitch": {
                "mean": float(np.mean(lang_mean_pitches)) if lang_mean_pitches else np.nan,
                "median": float(np.median(lang_median_pitches)) if lang_median_pitches else np.nan,
                "std": float(np.std(lang_mean_pitches)) if lang_mean_pitches else np.nan,
                "min": float(np.min(lang_mean_pitches)) if lang_mean_pitches else np.nan,
                "max": float(np.max(lang_mean_pitches)) if lang_mean_pitches else np.nan
            },
            "speech_rate": {
                "mean_wpm": float(np.mean(lang_wpms)) if lang_wpms else np.nan,
                "median_wpm": float(np.median(lang_wpms)) if lang_wpms else np.nan,
                "std_wpm": float(np.std(lang_wpms)) if lang_wpms else np.nan,
                "mean_cps": float(np.mean(lang_cpss)) if lang_cpss else np.nan,
                "slow_count": lang_slow_count,
                "fast_count": lang_fast_count,
                "slow_percentage": (lang_slow_count / len(lang_speech_stats) * 100) if lang_speech_stats else 0.0,
                "fast_percentage": (lang_fast_count / len(lang_speech_stats) * 100) if lang_speech_stats else 0.0,
                "normal_percentage": ((len(lang_speech_stats) - lang_slow_count - lang_fast_count) / len(lang_speech_stats) * 100) if lang_speech_stats else 0.0
            }
        }
    
    return {
        "total_items": total_items,
        "text_lens": all_text_lens,
        "audio_lens": all_audio_lens,
        "snrs": all_snrs,
        "domains": all_domains,
        "languages": all_languages,
        "word_counts": combined_word_counts,
        "text_vs_durs": combined_text_vs_durs,
        "text_len_counter": text_len_counter,
        "domain_counter": domain_counter,
        "language_counter": language_counter,
        "avg_snr": avg_snr,
        "median_snr": median_snr,
        "std_snr": std_snr,
        "min_snr": np.min(all_snrs) if all_snrs else np.nan,
        "max_snr": np.max(all_snrs) if all_snrs else np.nan,
        "pitch_stats": all_pitch_stats,
        "speech_rate_stats": all_speech_rate_stats,
        "mean_pitches": valid_mean_pitches,
        "median_pitches": valid_median_pitches,
        "avg_wpm": avg_wpm,
        "median_wpm": median_wpm,
        "std_wpm": std_wpm,
        "avg_cps": avg_cps,
        "total_slow_count": total_slow_count,
        "total_fast_count": total_fast_count,
        "total_clipped_count": total_clipped_count,
        "total_usable_for_medical": total_usable_count,
        "language_wise_stats": language_wise_stats,
        "quality_assessments": all_quality_assessments
    }


def convert_numpy_types(obj):
    """Convert NumPy types to native Python types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, Counter):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    return obj


def save_results(aggregated_stats: Dict, output_dir: str):
    """Save analysis results to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    domains_dict = {k: int(v) for k, v in aggregated_stats["domain_counter"].items()}
    languages_dict = {k: int(v) for k, v in aggregated_stats["language_counter"].items()}
    
    summary = {
        "total_items": int(aggregated_stats["total_items"]),
        "usable_for_medical": int(aggregated_stats["total_usable_for_medical"]),
        "medical_usable_percentage": (aggregated_stats["total_usable_for_medical"] / aggregated_stats["total_items"] * 100) if aggregated_stats["total_items"] > 0 else 0.0,
        "domains": domains_dict,
        "languages": languages_dict,
        "audio_statistics": {
            "mean_duration": float(np.mean(aggregated_stats["audio_lens"])) if aggregated_stats["audio_lens"] else 0.0,
            "median_duration": float(np.median(aggregated_stats["audio_lens"])) if aggregated_stats["audio_lens"] else 0.0,
            "std_duration": float(np.std(aggregated_stats["audio_lens"])) if aggregated_stats["audio_lens"] else 0.0,
            "min_duration": float(np.min(aggregated_stats["audio_lens"])) if aggregated_stats["audio_lens"] else 0.0,
            "max_duration": float(np.max(aggregated_stats["audio_lens"])) if aggregated_stats["audio_lens"] else 0.0
        },
        "text_statistics": {
            "mean_length": float(np.mean(aggregated_stats["text_lens"])) if aggregated_stats["text_lens"] else 0.0,
            "median_length": float(np.median(aggregated_stats["text_lens"])) if aggregated_stats["text_lens"] else 0.0,
            "std_length": float(np.std(aggregated_stats["text_lens"])) if aggregated_stats["text_lens"] else 0.0,
            "min_length": int(np.min(aggregated_stats["text_lens"])) if aggregated_stats["text_lens"] else 0,
            "max_length": int(np.max(aggregated_stats["text_lens"])) if aggregated_stats["text_lens"] else 0
        },
        "snr_statistics": {
            "mean_db": float(aggregated_stats["avg_snr"]) if not (np.isnan(aggregated_stats["avg_snr"]) or np.isinf(aggregated_stats["avg_snr"])) else None,
            "median_db": float(aggregated_stats["median_snr"]) if not (np.isnan(aggregated_stats["median_snr"]) or np.isinf(aggregated_stats["median_snr"])) else None,
            "std_db": float(aggregated_stats["std_snr"]) if not (np.isnan(aggregated_stats["std_snr"]) or np.isinf(aggregated_stats["std_snr"])) else None,
            "min_db": float(aggregated_stats["min_snr"]) if not (np.isnan(aggregated_stats["min_snr"]) or np.isinf(aggregated_stats["min_snr"])) else None,
            "max_db": float(aggregated_stats["max_snr"]) if not (np.isnan(aggregated_stats["max_snr"]) or np.isinf(aggregated_stats["max_snr"])) else None,
            "threshold_good": 25.0,
            "threshold_min": 15.0
        },
        "speech_rate": {
            "mean_wpm": float(aggregated_stats["avg_wpm"]) if not (np.isnan(aggregated_stats["avg_wpm"]) or np.isinf(aggregated_stats["avg_wpm"])) else None,
            "median_wpm": float(aggregated_stats["median_wpm"]) if not (np.isnan(aggregated_stats["median_wpm"]) or np.isinf(aggregated_stats["median_wpm"])) else None,
            "std_wpm": float(aggregated_stats["std_wpm"]) if not (np.isnan(aggregated_stats["std_wpm"]) or np.isinf(aggregated_stats["std_wpm"])) else None,
            "mean_cps": float(aggregated_stats["avg_cps"]) if not (np.isnan(aggregated_stats["avg_cps"]) or np.isinf(aggregated_stats["avg_cps"])) else None,
            "slow_count": int(aggregated_stats.get("total_slow_count", 0)),
            "fast_count": int(aggregated_stats.get("total_fast_count", 0)),
            "slow_percentage": (aggregated_stats.get("total_slow_count", 0) / aggregated_stats["total_items"] * 100) if aggregated_stats["total_items"] > 0 else 0.0,
            "fast_percentage": (aggregated_stats.get("total_fast_count", 0) / aggregated_stats["total_items"] * 100) if aggregated_stats["total_items"] > 0 else 0.0
        },
        "pitch_statistics": {
            "mean_pitch_hz": float(np.mean(aggregated_stats["mean_pitches"])) if aggregated_stats.get("mean_pitches") and len(aggregated_stats["mean_pitches"]) > 0 else None,
            "median_pitch_hz": float(np.median(aggregated_stats["median_pitches"])) if aggregated_stats.get("median_pitches") and len(aggregated_stats["median_pitches"]) > 0 else None,
            "std_pitch_hz": float(np.std(aggregated_stats["mean_pitches"])) if aggregated_stats.get("mean_pitches") and len(aggregated_stats["mean_pitches"]) > 0 else None,
            "min_pitch_hz": float(np.min(aggregated_stats["mean_pitches"])) if aggregated_stats.get("mean_pitches") and len(aggregated_stats["mean_pitches"]) > 0 else None,
            "max_pitch_hz": float(np.max(aggregated_stats["mean_pitches"])) if aggregated_stats.get("mean_pitches") and len(aggregated_stats["mean_pitches"]) > 0 else None
        },
        "audio_quality": {
            "clipped_count": int(aggregated_stats.get("total_clipped_count", 0)),
            "clipped_percentage": (aggregated_stats.get("total_clipped_count", 0) / aggregated_stats["total_items"] * 100) if aggregated_stats["total_items"] > 0 else 0.0
        },
        "vocabulary_size": int(len(aggregated_stats["word_counts"])),
        "language_wise_statistics": aggregated_stats.get("language_wise_stats", {})
    }
    
    summary = convert_numpy_types(summary)
    
    summary_file = os.path.join(output_dir, "medical_tts_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Saved summary to: {summary_file}")
    
    if aggregated_stats["word_counts"]:
        word_df = pd.DataFrame.from_dict(aggregated_stats["word_counts"], orient='index', columns=['count'])
        word_df.sort_values('count', ascending=False, inplace=True)
        word_file = os.path.join(output_dir, "word_frequencies.csv")
        word_df.to_csv(word_file, encoding='utf-8')
        print(f"✅ Saved word frequencies to: {word_file}")
    
    if aggregated_stats["domain_counter"]:
        domain_df = pd.DataFrame.from_dict(aggregated_stats["domain_counter"], orient='index', columns=['count'])
        domain_df.sort_values('count', ascending=False, inplace=True)
        domain_file = os.path.join(output_dir, "domain_distribution.csv")
        domain_df.to_csv(domain_file, encoding='utf-8')
        print(f"✅ Saved domain distribution to: {domain_file}")
    
    if aggregated_stats["language_counter"]:
        lang_df = pd.DataFrame.from_dict(aggregated_stats["language_counter"], orient='index', columns=['count'])
        lang_df.sort_values('count', ascending=False, inplace=True)
        lang_file = os.path.join(output_dir, "language_distribution.csv")
        lang_df.to_csv(lang_file, encoding='utf-8')
        print(f"✅ Saved language distribution to: {lang_file}")
    
    if aggregated_stats.get("language_wise_stats"):
        lang_stats_data = []
        for lang, stats in aggregated_stats["language_wise_stats"].items():
            lang_stats_data.append({
                "language": lang,
                "count": stats["count"],
                "snr_mean_db": stats["snr"]["mean"] if not np.isnan(stats["snr"]["mean"]) else None,
                "snr_median_db": stats["snr"]["median"] if not np.isnan(stats["snr"]["median"]) else None,
                "snr_min_db": stats["snr"]["min"] if not np.isnan(stats["snr"]["min"]) else None,
                "snr_max_db": stats["snr"]["max"] if not np.isnan(stats["snr"]["max"]) else None,
                "pitch_mean_hz": stats["pitch"]["mean"] if not np.isnan(stats["pitch"]["mean"]) else None,
                "pitch_median_hz": stats["pitch"]["median"] if not np.isnan(stats["pitch"]["median"]) else None,
                "pitch_std_hz": stats["pitch"]["std"] if not np.isnan(stats["pitch"]["std"]) else None,
                "pitch_range_hz": (stats["pitch"]["max"] - stats["pitch"]["min"]) if not (np.isnan(stats["pitch"]["max"]) or np.isnan(stats["pitch"]["min"])) else None,
                "speech_rate_mean_wpm": stats["speech_rate"]["mean_wpm"] if not np.isnan(stats["speech_rate"]["mean_wpm"]) else None,
                "speech_rate_median_wpm": stats["speech_rate"]["median_wpm"] if not np.isnan(stats["speech_rate"]["median_wpm"]) else None,
                "speech_rate_mean_cps": stats["speech_rate"]["mean_cps"] if not np.isnan(stats["speech_rate"]["mean_cps"]) else None,
                "slow_count": stats["speech_rate"]["slow_count"],
                "fast_count": stats["speech_rate"]["fast_count"],
                "slow_percentage": stats["speech_rate"]["slow_percentage"],
                "fast_percentage": stats["speech_rate"]["fast_percentage"],
                "normal_percentage": stats["speech_rate"]["normal_percentage"]
            })
        
        lang_stats_df = pd.DataFrame(lang_stats_data)
        lang_stats_file = os.path.join(output_dir, "language_wise_statistics.csv")
        lang_stats_df.to_csv(lang_stats_file, index=False, encoding='utf-8')
        print(f"✅ Saved language-wise statistics to: {lang_stats_file}")
    
    print("\n" + "="*70)
    print("MEDICAL TTS DATASET ANALYSIS SUMMARY")
    print("="*70)


def find_all_datasets(data_root: str) -> List[Path]:
    """Find all dataset directories containing metadata.csv."""
    datasets = []
    data_path = Path(data_root)
    
    if not data_path.exists():
        print(f"❌ Data root directory does not exist: {data_root}")
        return datasets
    
    for metadata_file in data_path.rglob("metadata.csv"):
        dataset_dir = metadata_file.parent
        datasets.append(dataset_dir)
    
    return sorted(datasets)


def main():
    """Main analysis function."""
    print("="*70)
    print("MEDICAL TTS DATASET EDA - REFACTORED")
    print("="*70)
    print(f"Data root: {DATA_ROOT}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Number of processes: {NUM_PROC}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*70)
    
    print("\n🔍 Searching for datasets...")
    datasets = find_all_datasets(DATA_ROOT)
    print(f"✅ Found {len(datasets)} dataset(s) with metadata.csv")
    
    if not datasets:
        print("❌ No datasets found. Exiting.")
        return
    
    print("\n📂 Loading metadata from all datasets...")
    all_items = []
    dataset_info = {}
    
    for dataset_path in datasets:
        dataset_name = dataset_path.name
        print(f"  Processing: {dataset_name}")
        items = formatter(str(dataset_path), "metadata.csv")
        if items:
            all_items.extend(items)
            dataset_info[dataset_name] = len(items)
            print(f"    ✅ Found {len(items)} items matching criteria")
        else:
            print(f"    ⚠️  No items matching criteria")
    
    print(f"\n✅ Total items to analyze: {len(all_items)}")
    
    if not all_items:
        print("❌ No items found matching HEALTH or LOCAL CONVERSATION domains. Exiting.")
        return
    
    print(f"\n🔄 Processing {len(all_items)} items in batches of {BATCH_SIZE}...")
    num_batches = (len(all_items) + BATCH_SIZE - 1) // BATCH_SIZE
    all_batch_stats = []
    
    for i in range(0, len(all_items), BATCH_SIZE):
        batch = all_items[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        batch_stats = process_batch(batch, batch_num, num_batches)
        if batch_stats:
            all_batch_stats.append(batch_stats)
        del batch
        gc.collect()
    
    print("\n📊 Aggregating statistics...")
    aggregated_stats = aggregate_statistics(all_batch_stats)
    
    print("\n💾 Saving results...")
    save_results(aggregated_stats, OUTPUT_DIR)
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()