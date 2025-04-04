import timm

print(timm.list_models(
    filter="tinynet*",
    pretrained=False))