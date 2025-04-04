import timm

print(timm.list_models(
    filter="focalnet*",
    pretrained=False))