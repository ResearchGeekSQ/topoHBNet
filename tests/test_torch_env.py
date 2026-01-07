import pytest
import torch
import numpy as np

def test_torch_installation():
    """Verify PyTorch is installed and importable."""
    print(f"\nPyTorch Version: {torch.__version__}")
    assert torch.__version__ is not None

def test_cuda_availability():
    """Verify CUDA is available and GPU can be accessed."""
    print(f"\nCUDA Available: {torch.cuda.is_available()}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Device Count: {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        print(f"Current Device: {torch.cuda.current_device()}")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        assert torch.cuda.is_available() is True
    else:
        pytest.skip("CUDA not available")

def test_torch_tensor_operations():
    """Verify basic tensor operations on GPU (if available)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nRunning tensor test on: {device}")
    
    x = torch.tensor([1.0, 2.0, 3.0]).to(device)
    y = torch.tensor([4.0, 5.0, 6.0]).to(device)
    z = x + y
    
    expected = np.array([5.0, 7.0, 9.0])
    np.testing.assert_allclose(z.cpu().numpy(), expected, rtol=1e-5)

def test_torch_scatter_import():
    """Verify torch-scatter is installed (required for TopoX)."""
    try:
        import torch_scatter
        print(f"\ntorch-scatter Version: {torch_scatter.__version__}")
        assert torch_scatter.__version__ is not None
    except ImportError:
        pytest.fail("torch-scatter is not installed")

if __name__ == "__main__":
    try:
        test_torch_installation()
        test_cuda_availability()
        test_torch_tensor_operations()
        test_torch_scatter_import()
        print("\nAll manual tests passed successfully!")
    except Exception as e:
        print(f"\nTest failed: {e}")
