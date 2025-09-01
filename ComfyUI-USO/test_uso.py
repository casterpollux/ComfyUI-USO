"""Test script for USO ComfyUI Node"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all imports work"""
    try:
        from uso_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
        print("✅ Imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_node_registration():
    """Test that nodes are properly registered"""
    from uso_nodes import NODE_CLASS_MAPPINGS
    
    required_nodes = ["USOModelLoader", "USOImageEncoder", "USOSampler", "USOLatentToImage"]
    for node in required_nodes:
        if node in NODE_CLASS_MAPPINGS:
            print(f"✅ {node} registered")
        else:
            print(f"❌ {node} not found")
            return False
    return True

def test_parameter_ranges():
    """Test that parameters match HF demo"""
    from uso_nodes import USOSampler
    
    inputs = USOSampler.INPUT_TYPES()
    
    # Check guidance range
    guidance = inputs["required"]["guidance"]
    assert guidance[1]["min"] == 1.0, "Guidance min should be 1.0"
    assert guidance[1]["max"] == 5.0, "Guidance max should be 5.0"
    assert guidance[1]["default"] == 4.0, "Guidance default should be 4.0"
    
    # Check steps range
    steps = inputs["required"]["steps"]
    assert steps[1]["default"] == 25, "Steps default should be 25"
    assert steps[1]["max"] == 50, "Steps max should be 50"
    assert steps[1]["min"] == 1, "Steps min should be 1"
    
    # Check dimensions
    width = inputs["required"]["width"]
    height = inputs["required"]["height"]
    assert width[1]["min"] == 512, "Width min should be 512"
    assert width[1]["max"] == 1536, "Width max should be 1536"
    assert height[1]["min"] == 512, "Height min should be 512"
    assert height[1]["max"] == 1536, "Height max should be 1536"
    
    # Check seed supports random
    seed = inputs["required"]["seed"]
    assert seed[1]["min"] == -1, "Seed should support -1 for random"
    assert seed[1]["default"] == -1, "Seed default should be -1"
    
    # Check content_reference_size exists
    content_ref_size = inputs["optional"]["content_reference_size"]
    assert content_ref_size[1]["default"] == 512, "Content reference size default should be 512"
    
    print("✅ Parameter ranges match HF demo")
    return True

def test_display_names():
    """Test that display names are properly set"""
    from uso_nodes import NODE_DISPLAY_NAME_MAPPINGS
    
    expected_names = {
        "USOModelLoader": "USO Model Loader",
        "USOImageEncoder": "USO Image Encoder", 
        "USOSampler": "USO Sampler",
        "USOLatentToImage": "USO Decode"
    }
    
    for node_id, expected_name in expected_names.items():
        if NODE_DISPLAY_NAME_MAPPINGS.get(node_id) == expected_name:
            print(f"✅ {node_id} display name correct")
        else:
            print(f"❌ {node_id} display name incorrect")
            return False
    return True

def test_inference_config():
    """Test that inference config matches HF demo"""
    from uso_inference import USOConfig
    
    config = USOConfig()
    assert config.num_inference_steps == 25, "Config steps should be 25"
    assert config.guidance_scale == 4.0, "Config guidance should be 4.0"
    assert config.content_reference_size == 512, "Config content_reference_size should be 512"
    
    print("✅ Inference config matches HF demo")
    return True

if __name__ == "__main__":
    print("Testing USO ComfyUI Node...")
    print("=" * 50)
    
    all_tests = [
        test_imports,
        test_node_registration,
        test_parameter_ranges,
        test_display_names,
        test_inference_config
    ]
    
    passed = 0
    failed = 0
    
    for test in all_tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed with error: {e}")
            failed += 1
        print()
    
    print("=" * 50)
    print(f"Tests passed: {passed}")
    print(f"Tests failed: {failed}")
    
    if failed == 0:
        print("\n🎉 All tests passed! USO node is ready.")
        exit(0)
    else:
        print(f"\n❌ {failed} tests failed. Check the errors above.")
        exit(1)

