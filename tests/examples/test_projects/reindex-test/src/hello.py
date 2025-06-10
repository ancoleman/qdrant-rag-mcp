def hello():
    """Say hello to the world."""
    return "Hello, World!"

def greet(name):
    """Greet someone by name."""
    return f"Hello, {name}!"

class Greeter:
    """A class for greeting people."""
    
    def __init__(self, default_greeting="Hello"):
        self.default_greeting = default_greeting
    
    def greet(self, name):
        return f"{self.default_greeting}, {name}!"