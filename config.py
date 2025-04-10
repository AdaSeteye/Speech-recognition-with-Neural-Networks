%%writefile config.yaml

subset: 1.0
learning_rate: 0.001
epochs: 100
train_beam_width: 3
test_beam_width: 10
mfcc_features: 28 
embed_size: 256   

batch_size: 64 

encoder dropout: 0.2
lstm dropout: 0.2
decoder dropout: 0.2
freq_mask_param: 4
time_mask_param: 8
augment:True


import yaml
with open("config.yaml") as file:
    config = yaml.safe_load(file)
