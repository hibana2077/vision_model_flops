import yaml
import json
import torch
import timm
from fvcore.nn import FlopCountAnalysis, flop_count_table
from pathlib import Path
import argparse

def load_config(config_path):
    """Load model configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def analyze_model(model_name, input_shape=(1, 3, 224, 224)):
    """Analyze a model from timm and return FLOPs, MACs, and parameters."""
    try:
        # Create model
        model = timm.create_model(model_name, pretrained=False)
        model.eval()
        
        # Create dummy input
        input_tensor = torch.rand(input_shape)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        
        # Analyze FLOPs
        flops = FlopCountAnalysis(model, input_tensor)
        
        # Get total FLOPs
        total_flops = flops.total()
        
        # Get FLOPs by operator
        flops_by_operator = dict(flops.by_operator())
        
        # Get FLOPs by module
        flops_by_module = dict(flops.by_module())
        
        # Calculate MACs (Multiply-Accumulate Operations)
        # MACs are approximately FLOPs/2 for most neural network operations
        total_macs = total_flops // 2
        
        return {
            "model_name": model_name,
            "total_params": total_params,
            "total_flops": total_flops,
            "total_macs": total_macs,
            "flops_by_operator": flops_by_operator,
            "flops_by_module": flops_by_module
        }
    except Exception as e:
        return {
            "model_name": model_name,
            "error": str(e)
        }

def main():
    parser = argparse.ArgumentParser(description="Analyze neural network models for FLOPs, MACs, and parameters")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
    parser.add_argument("--output", type=str, default="model_analysis.json", help="Path to the output JSON file")
    parser.add_argument("--input_shape", type=str, default="1,3,224,224", help="Input shape as comma-separated values")
    args = parser.parse_args()
    
    # Parse input shape
    input_shape = tuple(map(int, args.input_shape.split(',')))
    
    # Load configuration
    config = load_config(args.config)
    
    # Get model names from configuration
    model_names = config.get("models", [])
    if not model_names:
        print("No models specified in the configuration file.")
        return
    
    # Analyze each model
    results = []
    for model_name in model_names:
        print(f"Analyzing model: {model_name}")
        result = analyze_model(model_name, input_shape)
        results.append(result)
    
    # Save results to JSON file
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Analysis completed. Results saved to {output_path}")

if __name__ == "__main__":
    main()