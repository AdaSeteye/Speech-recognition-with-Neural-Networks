from architecture import ASRModel
import config
from imports import device, nn, torch, cuda_ctc_decoder, os
from dictionary import PHONEMES, LABELS
from utils import *
from data_loaders import train_loader, val_loader


model = ASRModel(
    input_size  = config['mfcc_features'],  
    embed_size  = config['embed_size'], 
    output_size = len(PHONEMES)
).to(device)


criterion = nn.CTCLoss(
    blank=0,
    reduction='mean',
    zero_infinity=True
)

optimizer =  torch.optim.AdamW(model.parameters(),lr=config['learning_rate']) 

decoder = cuda_ctc_decoder(tokens=LABELS, nbest=1, beam_size=config['train_beam_width'])

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
   optimizer,
   mode='min',
   factor=0.5,
   patience=3,
   verbose=True
)
scaler = torch.cuda.amp.GradScaler()

criterion = torch.nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])

decoder = cuda_ctc_decoder(
    tokens=LABELS, nbest=1, beam_size=config['train_beam_width']
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, verbose=True
)

scaler = torch.cuda.amp.GradScaler()


# Set up checkpoint directories and WanDB logging watch
checkpoint_root = os.path.join(os.getcwd(), 'checkpoints')
os.makedirs(checkpoint_root, exist_ok=True)

checkpoint_best_model_filename = 'checkpoint-best-model.pth'
checkpoint_last_epoch_filename = 'checkpoint-last-epoch.pth'
epoch_model_path = os.path.join(checkpoint_root, checkpoint_last_epoch_filename)
best_model_path = os.path.join(checkpoint_root, checkpoint_best_model_filename)



last_epoch_completed = 0
best_lev_dist = float("inf")


for epoch in range(last_epoch_completed, config['epochs']):
    print("\nEpoch: {}/{}".format(epoch + 1, config['epochs']))

    curr_lr = optimizer.param_groups[0]['lr']

    train_loss = train_model(model, train_loader, criterion, optimizer)

    valid_loss, valid_dist = validate_model(model, val_loader, decoder)

    scheduler.step(valid_dist)

    print("\tTrain Loss {:.04f}\t Learning Rate {:.07f}".format(train_loss, curr_lr))
    print("\tVal Dist {:.04f}\t Val Loss {:.04f}".format(valid_dist, valid_loss))

    

    # Save last epoch model
    save_model(model, optimizer, scheduler, ['valid_dist', valid_dist], epoch, epoch_model_path)
    
    if valid_dist <= best_lev_dist:
        best_lev_dist = valid_dist
        save_model(model, optimizer, scheduler, ['valid_dist', valid_dist], epoch, best_model_path)
        




    
         
   
