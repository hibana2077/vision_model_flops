import timm

print(timm.list_models(
    filter="inception*",
    pretrained=False))