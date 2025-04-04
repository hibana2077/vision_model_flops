import timm

print(timm.list_models(
    filter="hardcorenas*",
    pretrained=False))