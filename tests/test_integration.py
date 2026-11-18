"""
Integration Test for the complete Glaucoma Detection Pipeline.
"""

import os
import sys
import numpy as np
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.integration.inference import GlaucomaInferenceEngine

def test_inference_pipeline():
    engine = GlaucomaInferenceEngine()
    
    # Synthetic fundus image (H=512, W=512, C=3)
    dummy_img = np.random.randint(50, 200, (512, 512, 3), dtype=np.uint8)
    
    result = engine.predict(dummy_img)
    
    # Assert keys exist
    expected_keys = [
        'original_image', 'predicted_mask', 'mask_overlay',
        'optic_disc_mask', 'optic_cup_mask', 'glaucoma_probability',
        'prediction_label', 'gradcam_heatmap', 'gradcam_overlay',
        'clinical_features', 'disclaimer'
    ]
    for k in expected_keys:
        assert k in result
        
    assert 0.0 <= result['glaucoma_probability'] <= 1.0
    assert result['predicted_mask'].shape == (512, 512)
    assert result['mask_overlay'].shape == (512, 512, 3)
    assert result['gradcam_overlay'].shape == (512, 512, 3)
    assert 'vcdr' in result['clinical_features']

if __name__ == '__main__':
    pytest.main([__file__])
