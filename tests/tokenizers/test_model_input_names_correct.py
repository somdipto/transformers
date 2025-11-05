#!/usr/bin/env python3
"""
Test script to verify the model_input_names singleton fix.

This creates a minimal tokenizer class to test the fix without dependencies.
"""

import sys
import os

# Add the transformers source to the path
sys.path.insert(0, '/workspace/transformers/src')

# Import the base class
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, SpecialTokensMixin
from transformers.utils import PushToHubMixin

def test_model_input_names_singleton_fix():
    """Test that model_input_names is now instance-specific rather than shared."""
    print("Testing model_input_names singleton fix with mock tokenizer...")
    
    # Create a minimal tokenizer class for testing
    class MockTokenizer(PreTrainedTokenizerBase):
        pass
    
    # Test 1: Create first tokenizer instance
    print("\n1. Creating first mock tokenizer instance")
    tok1 = MockTokenizer()
    print(f"   tok1.model_input_names = {tok1.model_input_names}")
    
    # Verify initial state
    original_names = list(tok1.model_input_names)
    expected_names = ["input_ids", "token_type_ids", "attention_mask"]
    assert original_names == expected_names, f"Expected {expected_names}, got {original_names}"
    print("   ✓ Initial model_input_names looks correct")
    
    # Test 2: Modify the first instance using the setter
    print("\n2. Modifying tok1.model_input_names using the setter")
    new_names = original_names + ["CUSTOM_INPUT"]
    tok1.model_input_names = new_names
    print(f"   tok1.model_input_names = {tok1.model_input_names}")
    
    # Test 3: Create second tokenizer instance
    print("\n3. Creating second mock tokenizer instance")
    tok2 = MockTokenizer()
    print(f"   tok2.model_input_names = {tok2.model_input_names}")
    
    # Test 4: Verify the fix
    print("\n4. Verifying the fix:")
    
    # tok1 should have the custom input
    tok1_names = list(tok1.model_input_names)
    print(f"   tok1.model_input_names = {tok1_names}")
    assert "CUSTOM_INPUT" in tok1_names, "tok1 should have the custom input"
    print("   ✓ tok1 has the custom input as expected")
    
    # tok2 should NOT have the custom input (this was the original bug!)
    tok2_names = list(tok2.model_input_names)
    print(f"   tok2.model_input_names = {tok2_names}")
    assert "CUSTOM_INPUT" not in tok2_names, "tok2 should NOT have the custom input (this was the bug!)"
    assert tok2_names == expected_names, f"tok2 should have original names {expected_names}, got {tok2_names}"
    print("   ✓ tok2 has original names (fix works!)")
    
    # Verify they are truly different objects (not just same content)
    assert tok1_names != tok2_names, "The two instances should have different model_input_names"
    print("   ✓ tok1 and tok2 have different model_input_names")
    
    # Test 5: Verify external mutations don't affect the tokenizer
    print("\n5. Testing that external mutations to returned list don't affect internal state")
    tok3 = MockTokenizer()
    external_list = tok3.model_input_names  # This returns a copy
    external_list.append("EXTERNAL_MUTATION")
    
    # The tokenizer should not be affected by external mutations
    tok3_names = list(tok3.model_input_names)
    assert "EXTERNAL_MUTATION" not in tok3_names, "External mutations should not affect the tokenizer"
    print("   ✓ External mutations to returned list don't affect internal state")
    
    # Test 6: Multiple instances with different configurations
    print("\n6. Testing multiple instances with different configurations")
    tok4 = MockTokenizer(model_input_names=["input_ids", "custom_field"])
    tok5 = MockTokenizer(model_input_names=["input_ids", "token_type_ids", "attention_mask", "extra_field"])
    
    tok4_names = list(tok4.model_input_names)
    tok5_names = list(tok5.model_input_names)
    
    assert tok4_names == ["input_ids", "custom_field"], f"tok4 should have custom names, got {tok4_names}"
    assert tok5_names == ["input_ids", "token_type_ids", "attention_mask", "extra_field"], f"tok5 should have extra names, got {tok5_names}"
    print("   ✓ Different instances can have different model_input_names configurations")
    
    print("\n✅ All tests passed! The model_input_names singleton fix is working correctly.")
    return True

if __name__ == "__main__":
    try:
        test_model_input_names_singleton_fix()
        print("\n🎉 SUCCESS: The fix resolves the singleton issue!")
        print("\n📝 Summary:")
        print("   • Each tokenizer instance has its own independent model_input_names")
        print("   • Modifying one instance does not affect others")
        print("   • External mutations to returned lists are prevented")
        print("   • Custom configurations work correctly")
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()