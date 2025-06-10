def goodbye():
    """Say goodbye."""
    return "Goodbye!"

def farewell(name):
    """Say farewell to someone."""
    return f"Farewell, {name}!"

class FarewellManager:
    """Manages different types of farewells."""
    
    def __init__(self):
        self.farewells = ["Goodbye", "Farewell", "See you later", "Adios"]
    
    def random_farewell(self):
        import random
        return random.choice(self.farewells)