import timm

print(timm.list_models(
    filter="nfnet*",
    pretrained=False))