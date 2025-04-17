def get_model(model_name, args):
    name = model_name.lower()
    if name == "l2p":
        from models.l2p import Learner
    elif name == "dualprompt":
        from models.dualprompt import Learner
    elif name == "finetune":
        from models.finetune import Learner
    elif name == "der":
        from models.der import Learner
    elif name == "foster":
        from models.foster import Learner
    elif name == 'ranpac':
        from models.ranpac import Learner
    else:
        assert 0
    
    return Learner(args)
