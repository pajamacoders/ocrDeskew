import mlflow

class MLLogger:
    def __init__(self, config, logger):
        self.logger=logger
        self.cfg = config['mllogger_cfg']
        exp_name = self.cfg['exp_name']
        run_name = self.cfg['run_name']
        if "save_model_at_epoch" not in self.cfg.keys():
            self.cfg["save_model_at_epoch"]=[800]
        
        experiment = mlflow.get_experiment_by_name(exp_name)
        if not experiment:
            self.logger.info(f'No such experiment, Create experiment {exp_name} ')
            experiment_id = mlflow.create_experiment('ocr_deskew')
            experiment = mlflow.get_experiment(experiment_id)
        self.run = mlflow.start_run(run_name = run_name, experiment_id=experiment.experiment_id)
        self.logger.info('start run')
        for k,v in config.items():
            self.log_param(key=k, value=v)
        mlflow.set_tag("release.version", self.cfg['version'])
        
    def __del__(self):
        mlflow.end_run()

    def log_metric(self, key, value, step):
        mlflow.log_metric(key=key, value=value, step=step)
    
    def log_param(self, key, value):
        mlflow.log_param(key=key, value=value)

    def log_image(self, img, name):
        # numpy image numby uint8 dtype
        mlflow.log_image(img, artifact_file=f'test/{name}')

    def load_state_dict(self, path):
        pass

    def load_model(self, path):
        pass

    def log_model(self, jit_model, name):
        mlflow.pytorch.log_model(pytorch_mode=jit_model, artifact_path = f'model/{name}')

    def log_state_dict(self, epoch, model, optimizer=None, scheduler=None, isbest=False):
        if isbest:
            chkp_name=f'best_model'
        elif epoch in self.cfg['save_model_at_epoch']:
            chkp_name=f'{epoch}_epoch_chkpoint'
        else:
            chkp_name='latest_model'

        state_dict = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch
        }     

        mlflow.pytorch.log_state_dict(state_dict=state_dict, artifact_path=f'state_dicts/{chkp_name}')
