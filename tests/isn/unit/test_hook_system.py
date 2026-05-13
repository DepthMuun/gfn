import torch
import sys
from pathlib import Path

# Add project root
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from gfn.realizations.isn.hooks import ISNHook, HookManager

class MockHook(ISNHook):
    def __init__(self):
        self.scanner_called = False
        self.world_called = False

    def before_scanner(self, input_ids):
        self.scanner_called = True

    def before_world(self, world_input):
        self.world_called = True

def test_hook_manager_flow():
    """Verificar que el HookManager dispara los eventos esperados."""
    print("Testing Hook System Flow...")
    
    hook = MockHook()
    manager = HookManager([hook])
    
    # Simular flujo
    manager.before_scanner(torch.tensor([1, 2, 3]))
    assert hook.scanner_called, "before_scanner hook not triggered"
    
    manager.before_world({'impulses': torch.randn(1, 4, 128)})
    assert hook.world_called, "before_world hook not triggered"
    
    print("✓ Hook manager flow test passed")

if __name__ == "__main__":
    test_hook_manager_flow()
    print("\n[SUCCESS] Hook system unit tests passed.")
