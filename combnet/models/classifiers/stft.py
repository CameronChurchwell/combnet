import torch

n_fft = 256
chunk = 64
chunk_start = 20
hop_length = n_fft//8

class STFTClassifier(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.layers = torch.nn.Sequential(
            # torch.nn.Conv2d(1, 16, kernel_size=(n_fft, 16), stride=(1, 2), padding=(0, 1)),
            # torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1),
            torch.nn.Flatten(1, 2),
            torch.nn.Linear(n_fft//2+1, 2),
            # torch.nn.Linear(chunk, 2),
            torch.nn.Softmax(1)
        )

    def _convert_to_spec(self, x):
        # reshape
        x_flat = x.view(x.shape[0]*x.shape[1], x.shape[2])
        # stft
        s = torch.stft(x_flat, n_fft, hop_length=hop_length, return_complex=True, window=torch.hann_window(n_fft, device=x.device))
        # reshape back
        s = s.view(x.shape[0], x.shape[1], s.shape[-2], s.shape[-1])
        return abs(s)

    def forward(self, x):
        #convert to spectrogram
        spec = self._convert_to_spec(x) 
        # spec = spec[:, 0, chunk_start:chunk_start+chunk, :]
        spec = spec[:, 0]
        return self.layers(spec)