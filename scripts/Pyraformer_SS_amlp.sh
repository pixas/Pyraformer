python single_step_main.py -data_path data/elect/ -dataset elect -attn_type amlp -amlp_dim 32 -amlp_fn relu;
python single_step_main.py -data_path data/wind/ -dataset wind -attn_type amlp -amlp_dim 32 -amlp_fn relu;
python single_step_main.py -data_path data/flow/ -dataset flow -attn_type amlp -amlp_dim 32 -amlp_fn relu;