from imports import cuda_ctc_decoder, tqdm, device, torch, pd
from train import model
import config
from dictionary import LABELS
from utils import decode_prediction
from data_loaders import test_loader


test_decoder = cuda_ctc_decoder(tokens=LABELS, nbest=1, beam_size=config['test_beam_width'])

results = []

model.eval()
print("Testing")

for data in tqdm(test_loader):

    x, lx   = data
    x, lx   = x.to(device), lx.to(device)

    with torch.no_grad():
        h, lh = model(x, lx)

    prediction_string = decode_prediction(h, lh.to(device), test_decoder, LABELS)

    
    results.extend(prediction_string)

    del x, lx, h, lh
    torch.cuda.empty_cache()


if results:
    df = pd.DataFrame({
        'index': range(len(results)), 'label': results
    })

data_dir = "result.csv"
df.to_csv(data_dir, index = False)