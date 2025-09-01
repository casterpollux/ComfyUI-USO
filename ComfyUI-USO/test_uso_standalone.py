"""Standalone test script for USO ComfyUI Node (runs without ComfyUI environment)"""
import sys
import os

def test_file_structure():
    """Test that all required files exist"""
    required_files = [
        "__init__.py",
        "uso_nodes.py", 
        "uso_inference.py",
        "requirements.txt",
        "README.md",
        "examples/uso_workflow.json"
    ]
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    for file_path in required_files:
        full_path = os.path.join(base_dir, file_path)
        if os.path.exists(full_path):
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} missing")
            return False
    return True

def test_parameter_definitions():
    """Test parameter definitions in uso_nodes.py without importing"""
    uso_nodes_path = os.path.join(os.path.dirname(__file__), "uso_nodes.py")
    
    with open(uso_nodes_path, 'r') as f:
        content = f.read()
    
    # Check for HF demo parameter matches
    checks = [
        ('default": 25', "Steps default should be 25"),
        ('default": 4.0', "Guidance default should be 4.0"), 
        ('default": -1', "Seed default should be -1 for random"),
        ('min": 1,', "Steps min should be 1"),
        ('max": 50', "Steps max should be 50"),
        ('min": 1.0,', "Guidance min should be 1.0"),
        ('max": 5.0', "Guidance max should be 5.0"),
        ('min": 512', "Dimensions min should be 512"),
        ('max": 1536', "Dimensions max should be 1536"),
        ('content_reference_size', "Should have content_reference_size parameter"),
        ('"guidance":', "Should use 'guidance' not 'cfg'"),
    ]
    
    for check, description in checks:
        if check in content:
            print(f"✅ {description}")
        else:
            print(f"❌ {description}")
            return False
    return True

def test_config_definitions():
    """Test config definitions in uso_inference.py"""
    uso_inference_path = os.path.join(os.path.dirname(__file__), "uso_inference.py")
    
    with open(uso_inference_path, 'r') as f:
        content = f.read()
    
    checks = [
        ('num_inference_steps: int = 25', "Config steps should be 25"),
        ('guidance_scale: float = 4.0', "Config guidance should be 4.0"),
        ('content_reference_size: int = 512', "Config should have content_reference_size"),
        ('preprocess_reference_image', "Should have image preprocessing function"),
    ]
    
    for check, description in checks:
        if check in content:
            print(f"✅ {description}")
        else:
            print(f"❌ {description}")
            return False
    return True

def test_node_mappings():
    """Test that node mappings are properly defined"""
    uso_nodes_path = os.path.join(os.path.dirname(__file__), "uso_nodes.py")
    
    with open(uso_nodes_path, 'r') as f:
        content = f.read()
    
    required_nodes = [
        "USOModelLoader",
        "USOImageEncoder", 
        "USOSampler",
        "USOLatentToImage"
    ]
    
    for node in required_nodes:
        if f'"{node}": {node}' in content:
            print(f"✅ {node} properly mapped")
        else:
            print(f"❌ {node} not properly mapped")
            return False
    return True

def test_error_handling():
    """Test that error handling is present"""
    uso_nodes_path = os.path.join(os.path.dirname(__file__), "uso_nodes.py")
    
    with open(uso_nodes_path, 'r') as f:
        content = f.read()
    
    error_checks = [
        ('try:', "Should have try-catch blocks"),
        ('except FileNotFoundError', "Should handle missing files"),
        ('except torch.cuda.OutOfMemoryError', "Should handle OOM errors"),
        ('optimize_memory()', "Should have memory optimization"),
        ('RuntimeError', "Should raise user-friendly errors")
    ]
    
    for check, description in error_checks:
        if check in content:
            print(f"✅ {description}")
        else:
            print(f"❌ {description}")
            return False
    return True

def test_workflow_json():
    """Test that workflow JSON is valid"""
    import json
    
    workflow_path = os.path.join(os.path.dirname(__file__), "examples", "uso_workflow.json")
    
    try:
        with open(workflow_path, 'r') as f:
            workflow = json.load(f)
        
        # Check for USO nodes in workflow
        uso_nodes = ["USOModelLoader", "USOImageEncoder", "USOSampler", "USOLatentToImage"]
        found_nodes = []
        
        for node in workflow.get("nodes", []):
            if node.get("type") in uso_nodes:
                found_nodes.append(node.get("type"))
        
        if len(found_nodes) >= 3:  # At least 3 USO nodes
            print("✅ Workflow contains USO nodes")
            return True
        else:
            print(f"❌ Workflow missing USO nodes, found: {found_nodes}")
            return False
            
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in workflow: {e}")
        return False
    except Exception as e:
        print(f"❌ Error reading workflow: {e}")
        return False

if __name__ == "__main__":
    print("Testing USO ComfyUI Node (Standalone)")
    print("=" * 50)
    
    all_tests = [
        test_file_structure,
        test_parameter_definitions,
        test_config_definitions, 
        test_node_mappings,
        test_error_handling,
        test_workflow_json
    ]
    
    passed = 0
    failed = 0
    
    for test in all_tests:
        print(f"\nRunning {test.__name__}...")
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed with error: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"Tests passed: {passed}")
    print(f"Tests failed: {failed}")
    
    if failed == 0:
        print("\n🎉 All tests passed! USO node structure is correct.")
        print("\n📋 Next steps:")
        print("1. Copy ComfyUI-USO folder to your ComfyUI/custom_nodes/ directory")
        print("2. Install dependencies: pip install -r requirements.txt")
        print("3. Restart ComfyUI")
        print("4. Look for 'USO' category in the node menu")
        exit(0)
    else:
        print(f"\n❌ {failed} tests failed. Check the errors above.")
        exit(1)
