python single_step_main.py -data_path data/elect/ -dataset elect -enc_attn_type amlp -enc_amlp_dim 32 -enc_amlp_fn relu;
python single_step_main.py -data_path data/wind/ -dataset wind -enc_attn_type amlp -enc_amlp_dim 32 -enc_amlp_fn relu;
python single_step_main.py -data_path data/flow/ -dataset flow -enc_attn_type amlp -enc_amlp_dim 32 -enc_amlp_fn relu;