# Mock torch module to avoid import errors
class MockTorch:
    @staticmethod
    def tensor(data, dtype=None):
        return data
    
    @staticmethod
    def no_grad():
        return MockContext()
    
    class nn:
        @staticmethod
        def Linear(*args, **kwargs):
            return MockLinear()
        
        class Module:
            def __init__(self):
                pass

class MockContext:
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass

class MockLinear:
    def __call__(self, x):
        return x

import sys
sys.modules['torch'] = MockTorch()