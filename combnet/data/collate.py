import torch


###############################################################################
# Dataloader
###############################################################################


class Collate():

    def __init__(self, features=['audio']):
        self.features = features

    def __call__(self, batch):
        batch_values = []
        batch_size = len(batch)
        for feature, values in zip(self.features, zip(*batch)):

            # Pack audio
            if feature == 'audio':
                max_length = max([audio.shape[-1] for audio in values])
                padded_audio = torch.zeros(
                (batch_size, 1, max_length),
                dtype=torch.float)
                for i, audio in enumerate(values):
                    padded_audio[i, 0, :audio.shape[-1]] = audio[0]
                batch_values.append(padded_audio)

            # Pack stem
            elif feature == 'stem':
                batch_values.append(values)

            # Pack filename
            elif feature == 'audio_file':
                batch_values.append(values)

            # Pack lengths
            elif feature == 'length':
                batch_values.append(torch.tensor(values))

            # Pack input audio representation
            #TODO this is weird, values will always be a tuple...
                # the old code is wrong... Strange.
            elif isinstance(values[0], torch.Tensor):
                if values[0].dim() == 0: #assume all of them will have dim()==0
                    batch_values.append(torch.tensor(values))
                else:
                    max_length = max([latent.shape[-1] for latent in values])
                    padded_latents = torch.zeros(
                        (batch_size,) + values[0].shape[:-1] + (max_length,),
                        dtype=values[0].dtype)
                    for i, latent in enumerate(values):
                        padded_latents[i, ..., :latent.shape[-1]] = latent
                    batch_values.append(padded_latents)

            # Catch-all for other kinds
            else:
                batch_values.append(values)

        return batch_values