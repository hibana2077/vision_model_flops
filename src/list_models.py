import timm

print(timm.list_models(
    filter="mobilenetv*",
    pretrained=False))