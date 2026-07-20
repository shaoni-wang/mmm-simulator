# config.py
# Configuration for ABM model

class Config:
    """Configuration class for ABM parameters"""
    
    # Model parameters
    NUM_CONSUMERS = 200
    AVG_CONNECTIONS = 6
    INIT_ORANGE_PCT = 50
    
    # Product qualities
    QUALITY_ORANGE = 0.45
    QUALITY_BLUE = 0.50
    
    # Agent settings
    NORM_INFLUENCE = 0.50
    INFO_EXCHANGE = 0.50
    EXPLORATION = 0.01
    
    # Simulation settings
    SEED = 42
    NUM_STEPS = 100
    
    # World settings
    WORLD_WIDTH = 850
    WORLD_HEIGHT = 450
    WORLD_MARGIN = 40
    
    @classmethod
    def get_quality_list(cls):
        return [cls.QUALITY_ORANGE, cls.QUALITY_BLUE]