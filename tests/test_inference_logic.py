
import unittest
from unittest.mock import MagicMock, patch
import torch
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference_logic import run_inference_segmented, adaptive_duration_scale

class TestInferenceLogic(unittest.TestCase):

    def setUp(self):
        # Mock resources
        self.mock_model = MagicMock()
        self.mock_cfg = MagicMock()
        self.mock_cfg.y_sep_token = 123
        self.mock_cfg.x_sep_token = 124
        self.mock_cfg.add_eos_to_text = 0
        self.mock_cfg.add_bos_to_text = 0
        self.mock_cfg.eos = 1
        self.mock_cfg.eog = 1
        
        self.mock_text_tokenizer = MagicMock()
        self.mock_text_tokenizer.encode.return_value = [1, 2, 3] # Mock tokens
        self.mock_text_tokenizer.pad_token_id = 0
        
        self.mock_audio_tokenizer = MagicMock()
        self.mock_audio_tokenizer.device = "cpu"
        self.mock_audio_tokenizer.decode.return_value = torch.zeros(1, 16000) # Mock 1s audio

        self.resources = {
            "model": self.mock_model,
            "cfg": self.mock_cfg,
            "text_tokenizer": self.mock_text_tokenizer,
            "audio_tokenizer": self.mock_audio_tokenizer,
            "device": "cpu",
            "codec_audio_sr": 16000,
            "codec_sr": 50,
            "whisper_device": "cpu",
        }

    def test_adaptive_duration_scale(self):
        # Short duration
        self.assertAlmostEqual(adaptive_duration_scale(4.0, 1.5), 1.05)
        # Medium duration
        self.assertAlmostEqual(adaptive_duration_scale(12.5, 1.5), 1.25) # Midpoint
        # Long duration
        self.assertEqual(adaptive_duration_scale(25.0, 1.5), 1.5)

    @patch("inference_logic.inference_one_sample") 
    @patch("inference_logic.split_into_adaptive_chunks")
    @patch("inference_logic.save_audio")
    @patch("inference_logic.estimate_duration")
    @patch("inference_logic.normalize_text_with_lang")
    @patch("inference_logic.os.makedirs")
    @patch("inference_logic.get_audio_info")
    @patch("inference_logic.get_sample_rate")
    @patch("inference_logic.tokenize_audio") 
    def test_run_inference_segmented_flow(
        self,
        mock_tokenize_audio,
        mock_get_sample_rate,
        mock_get_audio_info,
        mock_makedirs,
        mock_normalize,
        mock_estimate, 
        mock_save_audio, 
        mock_split, 
        mock_inference_one
    ):
        # Setup mocks
        mock_split.return_value = ["This is a test chunk."]
        mock_normalize.return_value = ("This is a test chunk.", "en")
        mock_estimate.return_value = 5.0
        mock_get_sample_rate.return_value = 16000
        mock_tokenize_audio.return_value = torch.zeros(1, 1, 100) # dummy encoded frames
        
        # Mock inference return: (concat_sample, gen_sample, concat_frame, gen_frame)
        # We need frames to be tensors for eventual processing
        mock_frame = torch.zeros(1, 1, 100) # dummy frames
        mock_inference_one.return_value = (
            np.zeros(16000), # concat_sample
            np.zeros(16000), # gen_sample
            mock_frame,      # concat_frame
            mock_frame       # gen_frame
        )

        # Call function
        concatenated, chunk_results, chunks, first_seg_results, first_seg_texts, saved_files, all_segs = run_inference_segmented(
            reference_speech="dummy.wav",
            reference_text="Ref text",
            target_text="Target text",
            target_duration=None,
            top_k=50,
            top_p=0.9,
            min_p=0.0,
            temperature=1.0,
            seed=42,
            resources=self.resources,
            cut_off_sec=100,
            batch_count=1
        )

        # Assertions
        self.assertEqual(len(concatenated), 1) # 1 batch
        self.assertEqual(len(chunks), 1)
        mock_split.assert_called()
        mock_inference_one.assert_called()
        # Verify file saving was called (once for chunk, once for final)
        self.assertTrue(mock_save_audio.called)

if __name__ == "__main__":
    unittest.main()
