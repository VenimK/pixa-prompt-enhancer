"""
Unit tests for style functionality in the Pixa Prompt Enhancer application.
"""

import pytest
import sys
import os

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.style_constants import (
    STYLE_MAPPINGS, 
    FRONTEND_TO_BACKEND, 
    VALID_STYLE_KEYS, 
    VALID_DISPLAY_NAMES,
    get_backend_key,
    get_display_name,
    is_valid_style
)
from app.main import EnhanceRequest


class TestStyleConstants:
    """Test the style constants module."""
    
    def test_style_mappings_consistency(self):
        """Test that style mappings are consistent."""
        # Test that all backend keys are lowercase
        for backend_key in STYLE_MAPPINGS.keys():
            assert backend_key == backend_key.lower(), f"Backend key '{backend_key}' should be lowercase"
        
        # Test that reverse mapping is consistent
        for backend_key, display_name in STYLE_MAPPINGS.items():
            assert FRONTEND_TO_BACKEND[display_name] == backend_key, f"Reverse mapping inconsistent for '{backend_key}'"
    
    def test_valid_style_sets(self):
        """Test that valid style sets are consistent."""
        # Test that VALID_STYLE_KEYS matches STYLE_MAPPINGS keys
        assert VALID_STYLE_KEYS == set(STYLE_MAPPINGS.keys())
        
        # Test that VALID_DISPLAY_NAMES matches STYLE_MAPPINGS values
        assert VALID_DISPLAY_NAMES == set(STYLE_MAPPINGS.values())
    
    def test_get_backend_key(self):
        """Test backend key conversion."""
        # Test known style
        assert get_backend_key("Pixar 3D") == "pixar 3d"
        assert get_backend_key("Masters Of The Universe") == "masters of the universe"
        
        # Test unknown style (should return lowercase version)
        assert get_backend_key("Unknown Style") == "unknown style"
    
    def test_get_display_name(self):
        """Test display name conversion."""
        # Test known style
        assert get_display_name("pixar 3d") == "Pixar 3D"
        assert get_display_name("masters of the universe") == "Masters Of The Universe"
        
        # Test unknown style (should return title case)
        assert get_display_name("unknown style") == "Unknown Style"
    
    def test_is_valid_style(self):
        """Test style validation."""
        # Test valid backend keys
        assert is_valid_style("pixar 3d")
        assert is_valid_style("masters of the universe")
        
        # Test valid display names
        assert is_valid_style("Pixar 3D")
        assert is_valid_style("Masters Of The Universe")
        
        # Test invalid styles
        assert not is_valid_style("invalid style")
        assert not is_valid_style("")


class TestEnhanceRequestValidation:
    """Test the EnhanceRequest style validation."""
    
    def test_valid_style_validation(self):
        """Test that valid styles pass validation."""
        # Test valid backend keys
        request = EnhanceRequest(
            prompt="test prompt",
            prompt_type="Image",
            style="pixar 3d",
            cinematography="cinematic",
            lighting="natural"
        )
        assert request.style == "pixar 3d"
        
        # Test valid display names
        request = EnhanceRequest(
            prompt="test prompt",
            prompt_type="Image",
            style="Pixar 3D",
            cinematography="cinematic",
            lighting="natural"
        )
        assert request.style == "Pixar 3D"
    
    def test_empty_style_validation(self):
        """Test that empty/auto styles pass validation."""
        for style in ["", "none", "auto", "automatic"]:
            request = EnhanceRequest(
                prompt="test prompt",
                prompt_type="Image",
                style=style,
                cinematography="cinematic",
                lighting="natural"
            )
            assert request.style == style
    
    def test_invalid_style_validation(self):
        """Test that obviously invalid styles are handled."""
        # Test dangerous content
        with pytest.raises(ValueError, match="Style contains invalid content"):
            EnhanceRequest(
                prompt="test prompt",
                prompt_type="Image",
                style="<script>alert('xss')</script>",
                cinematography="cinematic",
                lighting="natural"
            )
        
        # Test too long style
        long_style = "x" * 101
        with pytest.raises(ValueError, match="Style name too long"):
            EnhanceRequest(
                prompt="test prompt",
                prompt_type="Image",
                style=long_style,
                cinematography="cinematic",
                lighting="natural"
            )
    
    def test_style_sanitization(self):
        """Test that style input is properly sanitized."""
        # Test whitespace trimming
        request = EnhanceRequest(
            prompt="test prompt",
            prompt_type="Image",
            style="  pixar 3d  ",
            cinematography="cinematic",
            lighting="natural"
        )
        assert request.style == "pixar 3d"


class TestNewStyles:
    """Test the newly added styles specifically."""
    
    def test_new_avatar_styles(self):
        """Test the new avatar styles are properly configured."""
        # Test backend keys exist
        assert "cute stylized 3d avatar" in VALID_STYLE_KEYS
        assert "real-to-avatar comparison" in VALID_STYLE_KEYS
        
        # Test display names exist
        assert "Cute Stylized 3D Avatar" in VALID_DISPLAY_NAMES
        assert "Real-to-Avatar Comparison" in VALID_DISPLAY_NAMES
        
        # Test mappings are correct
        assert get_backend_key("Cute Stylized 3D Avatar") == "cute stylized 3d avatar"
        assert get_backend_key("Real-to-Avatar Comparison") == "real-to-avatar comparison"
        assert get_display_name("cute stylized 3d avatar") == "Cute Stylized 3D Avatar"
        assert get_display_name("real-to-avatar comparison") == "Real-to-Avatar Comparison"
    
    def test_masters_of_universe_style(self):
        """Test the Masters of the Universe style is properly configured."""
        # Test backend key exists
        assert "masters of the universe" in VALID_STYLE_KEYS
        
        # Test display name exists
        assert "Masters Of The Universe" in VALID_DISPLAY_NAMES
        
        # Test mappings are correct
        assert get_backend_key("Masters Of The Universe") == "masters of the universe"
        assert get_display_name("masters of the universe") == "Masters Of The Universe"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
