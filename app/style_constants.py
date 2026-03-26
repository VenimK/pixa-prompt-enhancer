"""
Style constants for the Pixa Prompt Enhancer application.
This file maintains consistency between frontend and backend style definitions.
"""

# Style mappings - backend keys to frontend display names
STYLE_MAPPINGS = {
    # 3D & Animated Styles
    "3d pop-out pencil sketch": "3D Pop-Out Pencil Sketch",
    "pixar 3d": "Pixar 3D",
    "disney 3d": "Disney 3D",
    "cute stylized 3d avatar": "Cute Stylized 3D Avatar",
    "real-to-avatar comparison": "Real-to-Avatar Comparison",
    "dreamworks 3d": "DreamWorks 3D",
    "studio ghibli": "Studio Ghibli",
    "claymation": "Claymation",
    "stop motion": "Stop Motion",
    "low poly 3d": "Low Poly 3D",
    "isometric 3d": "Isometric 3D",
    "anime": "Anime",
    "cartoon": "Cartoon",
    "lego": "Lego",
    "minecraft": "Minecraft",
    "roblox": "Roblox",
    "pokemon": "Pokemon",
    "sonic": "Sonic",
    "mario": "Mario",
    "minions": "Minions",
    "fortnite": "Fortnite",
    "overwatch": "Overwatch",
    "genshin impact": "Genshin Impact",
    "animal crossing": "Animal Crossing",
    "among us": "Among Us",
    "illumination": "Illumination",
    "laika": "Laika",
    "aardman": "Aardman",
    "cartoon network": "Cartoon Network",
    "nickelodeon": "Nickelodeon",
    "league of legends": "League of Legends",
    "valorant": "Valorant",
    "apex legends": "Apex Legends",
    "halo": "Halo",
    "zelda": "Zelda",
    "final fantasy": "Final Fantasy",
    "dark souls": "Dark Souls",
    "street fighter": "Street Fighter",
    "arcane": "Arcane",
    "castlevania": "Castlevania",
    "cyberpunk edgerunners": "Cyberpunk Edgerunners",
    "funko pop": "Funko Pop",
    "transformers": "Transformers",
    "voxel art": "Voxel Art",
    "motion graphics": "Motion Graphics",
    
    # Pop Culture & Aesthetic Styles
    "vaporwave": "Vaporwave",
    "retrofuturism": "Retrofuturism",
    "dark academia": "Dark Academia",
    "cottagecore": "Cottagecore",
    "cyberpunk neon": "Cyberpunk Neon",
    "lo-fi aesthetic": "Lo-Fi Aesthetic",
    "solarpunk": "Solarpunk",
    "witchcore": "Witchcore",
    "masters of the universe": "Masters Of The Universe",
    
    # Add other styles as needed...
}

# Reverse mapping for frontend validation
FRONTEND_TO_BACKEND = {display: key for key, display in STYLE_MAPPINGS.items()}

# List of all valid style keys (for backend validation)
VALID_STYLE_KEYS = set(STYLE_MAPPINGS.keys())

# List of all valid display names (for frontend validation)
VALID_DISPLAY_NAMES = set(STYLE_MAPPINGS.values())

def get_backend_key(frontend_display_name: str) -> str:
    """Convert frontend display name to backend key."""
    return FRONTEND_TO_BACKEND.get(frontend_display_name, frontend_display_name.lower())

def get_display_name(backend_key: str) -> str:
    """Convert backend key to frontend display name."""
    return STYLE_MAPPINGS.get(backend_key, backend_key.title())

def is_valid_style(style_value: str) -> bool:
    """Check if a style value is valid (either backend key or display name)."""
    return style_value.lower() in VALID_STYLE_KEYS or style_value in VALID_DISPLAY_NAMES
