def runMiDaS(img, model: MiDaS):
    transform = midas_transforms.dpt_transform



    result = model(img)
    return result