import timm

# List all available models
print(timm.list_models(
    filter="volo*224",
    pretrained=False))