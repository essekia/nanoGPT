# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-bible'
eval_interval = 2500 # keep frequent because we'll overfit
eval_iters = 400
log_interval = 5 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'bible'
wandb_run_name = 'mini-gpt'

dataset = 'bible'
gradient_accumulation_steps = 1
batch_size = 24
block_size = 128 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 50
n_head = 50
n_embd = 1000
dropout = 0.1

learning_rate = 1e-4 # with baby networks can afford to go a bit higher
max_iters = 25000
lr_decay_iters = 5000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
