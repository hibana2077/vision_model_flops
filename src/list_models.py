import timm

print(timm.list_models(
    filter="mnasnet*",
    pretrained=False))