import os
import json
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision.utils import make_grid
from PIL import Image
import io

def load_model_analysis(json_path):
    """Load model analysis data from JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Filter out models with errors
    valid_models = [model for model in data if 'error' not in model]
    return valid_models

def create_bar_plot(models, metric_name, title, y_label, figsize=(8, 6)):
    """Create a bar plot for the specified metric"""
    plt.figure(figsize=figsize)
    
    # Extract model names and metric values
    model_names = [model['model_name'] for model in models]
    metric_values = [model[metric_name] for model in models]
    
    # For params, convert to millions
    if metric_name == 'total_params':
        metric_values = [val / 1_000_000 for val in metric_values]
        y_label = 'Parameters (M)'
    
    # Create bar plot
    bars = plt.bar(model_names, metric_values)
    
    # Add value labels on top of each bar
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2f}', ha='center', va='bottom', rotation=0)
    
    plt.title(title)
    plt.ylabel(y_label)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Convert to PIL Image
    plot_image = Image.open(buf)
    return plot_image

def fig_to_tensor(fig):
    """Convert a matplotlib figure to a PyTorch tensor"""
    # Save the figure to a BytesIO object
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    
    # Open as PIL Image and convert to tensor
    image = Image.open(buf)
    image_tensor = torchvision.transforms.ToTensor()(image)
    return image_tensor

def create_table_image(models, figsize=(10, 6)):
    """Create a table image with model information"""
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Hide axes
    ax.axis('off')
    
    # Set up table data
    table_data = []
    headers = ['Model', 'Params (M)', 'GFLOPs']
    table_data.append(headers)
    
    for model in models:
        model_name = model['model_name']
        params_m = model['total_params'] / 1_000_000
        gflops = model['gflops']
        
        table_data.append([model_name, f"{params_m:.2f}", f"{gflops:.2f}"])
    
    # Create the table
    table = ax.table(cellText=table_data[1:], colLabels=headers, 
                   loc='center', cellLoc='center')
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    # Save the table to a PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    table_image = Image.open(buf)
    plt.close(fig)
    
    return table_image

def main():
    # Path to the JSON file
    json_path = os.path.join('caformer', 'model_analysis.json')
    
    # Load model analysis data
    models = load_model_analysis(json_path)
    
    if not models:
        print("No valid models found in the JSON file.")
        return
    
    # Create plots
    params_plot = create_bar_plot(models, 'total_params', 'Model Parameters', 'Parameters (M)')
    gflops_plot = create_bar_plot(models, 'gflops', 'Model GFLOPs', 'GFLOPs')
    table_image = create_table_image(models)
    
    # Resize all images to the same size (use the size of the first image)
    target_size = params_plot.size
    params_plot = params_plot.resize(target_size)
    gflops_plot = gflops_plot.resize(target_size)
    table_image = table_image.resize(target_size)
    
    # Convert PIL images to tensors
    params_tensor = torchvision.transforms.ToTensor()(params_plot)
    gflops_tensor = torchvision.transforms.ToTensor()(gflops_plot)
    table_tensor = torchvision.transforms.ToTensor()(table_image)
    
    # Create a grid of images
    grid_tensors = torch.stack([params_tensor, gflops_tensor, table_tensor])
    grid = make_grid(grid_tensors, nrow=2, padding=20)
    # Convert grid tensor to PIL image
    grid_image = torchvision.transforms.ToPILImage()(grid)
    
    # Save the grid image
    output_path = 'model_analysis_grid.png'
    grid_image.save(output_path)
    print(f"Saved grid image to {output_path}")
    
    # Also save individual images
    params_plot.save('model_parameters.png')
    gflops_plot.save('model_gflops.png')
    table_image.save('model_table.png')
    print("Saved individual plots as well.")

if __name__ == "__main__":
    main()