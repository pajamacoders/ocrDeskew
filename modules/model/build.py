from .rotation_model import DeskewNet
models={
    "DeskewNet":DeskewNet
}
def build_model(type, args):
    return models[type](**args)
