import timm

print(timm.list_models(
    filter="edgenext*",
    pretrained=False))