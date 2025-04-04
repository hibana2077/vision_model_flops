import timm

print(timm.list_models(
    filter="convnext*",
    pretrained=False))