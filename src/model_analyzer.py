import yaml
import json
import torch
import timm
from thop import profile
from copy import deepcopy
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
        
        # Profile the model using thop
        macs, params = profile(deepcopy(model), inputs=(input_tensor,), verbose=False)
        
        # Calculate FLOPs (approximately 2x MACs)
        flops = macs * 2
        
        # Convert to billions for easier reading
        gflops = flops / 1E9
        gmacs = macs / 1E9
        
        return {
            "model_name": model_name,
            "total_params": int(params),
            "total_flops": int(flops),
            "total_macs": int(macs),
            "gflops": gflops,
            "gmacs": gmacs
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