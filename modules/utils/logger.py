import mlflow

class MLLogger:
    def __init__(self, config, logger):
        self.logger=logger
        
        experiment = mlflow.get_experiment_by_name('ocr_deskew')
        if not experiment:
            self.logger.info(f'No such experiment, Create experiment \'ocr_deskew\' ')
            experiment_id = mlflow.create_experiment('ocr_deskew')
            experiment = mlflow.get_experiment(experiment_id)
        
        run_name = config['run_name']
        self.run = mlflow.start_run(run_name = run_name, experiment_id=experiment.experiment_id)
        self.logger.info('start run')
        mlflow.set_tag("release.version", "0.1.0")
    def __del__(self):
        mlflow.end_run()

    def log_metric(self, key, value, step):
        mlflow.log_metric(key=key, value=value, step=step)
    
    def log_param(self, key, value):
        mlflow.log_param(key=key, value=value)

    def log_image(self, img, name):
        # numpy image numby uint8 dtype
        mlflow.log_image(img, artifact_file=f'train/{name}')

    def load_state_dict(self, path):
        pass

    def load_model(self, path):
        pass

    def log_model(self, jit_model, name):
        mlflow.pytorch.log_model(pytorch_mode=jit_model, artifact_path = f'model/{name}')

    def log_state_dict(self, model, optimizer, scheduler, loss, epoch, name, isbest=False):
        if isbest:
            chkp_name=f'best_model.chkp'
        elif epoch in [100,200,300]:
            chkp_name=f'{epoch}_{name}.chkp'
        else:
            chkp_name='latest_model.chkp'

        state_dict = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "loss": loss,
        }     

        mlflow.pytorch.log_state_dict(state_dict=state_dict, artifact_path=f'state_dicts/{chkp_name}')
