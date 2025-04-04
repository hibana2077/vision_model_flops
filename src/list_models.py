import timm

print(timm.list_models(
    filter="coat_lite*",
    pretrained=False))