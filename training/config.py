class cfg:
    
    model_name = 'resnext50_32x4d'
    
    seed = 16
    epochs = 25
    batch_size = 32
    
    
    label_dict = {'bgn'   : 0,
                  'NV'    : 1,
                  'mel'   : 2,
                  'bkl'   : 3,
                  'othr'  : 4,
                  'bcc'   : 5,
                  'akiec' : 6,
                  'vasc'  : 7,
                  'df'    : 8}