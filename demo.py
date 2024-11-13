from pipe.cfgs import load_cfg
from pipe.c2f_recons import Pipeline

cfg = load_cfg(f'pipe/cfgs/basic.yaml')
cfg.scene.input.rgb = 'data/1stb/0000000000.jpg'
vistadream = Pipeline(cfg)
vistadream()
