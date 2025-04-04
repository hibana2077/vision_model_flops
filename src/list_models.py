import timm

print(timm.list_models(
    filter="densenet*",
    pretrained=False))