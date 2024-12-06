import torch

# Simplify calling a convolution
def convolve( x, y):
    from torch.nn.functional import conv1d
    return conv1d( x[None,None], y[None,None], padding='same')[0,0]

# Take 1: Use an IIR comb filter, must use scan to avoid jit dealing with variable loop count
# Works fine, but has a DC peak and is insanely slow
def single_fractional_comb_iir( x, f0, a, sr):
    # Make the filter, essentially a fractional (sinc) delay in the feedback path
    l = sr/f0
    t = torch.arange( sr//20) - l + 1 # Fixed the buffer size to a constant to avoid jit complaints, I assume we don't care for sub 20Hz
    a = a * torch.sinc( t) * torch.exp( -.1 * t**2)
    # a = a.at[0].set( -.02)

    # Core scan routine utilizing a ring buffer
    c = torch.zeros( len( a))
    y = torch.zeros( len( x))
    for i in torch.arange( len( x)):
        y[i] = x[i] + a.dot( c)
        c = torch.roll( c, shifts=1)
        c[0] = y[i]

    # Do it
    return y

def single_comb_iir_faithful(x, f0, a, sr):
    if x.dim() == 2:
        x = x.unsqueeze(0)
    assert x.dim() == 3
    f0 = float(f0)
    sr = int(sr)
    l = int(round(sr/f0))
    y = x.clone()
    for i in range(l, x.shape[-1]):
        y[..., i] += a*y[..., i-l]
    return y

def single_comb_fir(x, f0, a, sr):
    if x.dim() == 2:
        x = x.unsqueeze(0)
    assert x.dim() == 3
    f0 = float(f0)
    sr = int(sr)
    l = int(round(sr/f0))
    y = x.clone()
    y[..., l:] += a*y[..., :-l]
    return y

def single_comb_fir_multitap(x, f0, a, sr):
    y = x
    if x.dim() == 2:
        x = x.unsqueeze(0)
    assert x.dim() == 3
    f0 = float(f0)
    sr = int(sr)
    l = int(round(sr/f0))
    y = x.clone()
    for i in range(1, 10):
        y[..., i*l:] += (a**i)*x[..., :-i*l]
    return y

def single_fractional_comb_fir_multitap(x, f0, a, sr):
    x = x.squeeze()
    l = sr/f0
    t = torch.arange(sr//10-1, -1, step=-1, device=x.device)

    f = torch.zeros(sr//10, device=x.device)
    f[-1] = 1.
    # TODO you can definitely remove this loop, but it might be less
    #  memory efficient and still not that much faster... The slow part is the conv
    for i in range(1, 11):
        f += (a ** i) * torch.sinc(t-i*l)
    # from matplotlib import pyplot as plt
    # plt.plot(f); plt.gcf().set_size_inches(10, 7.5); plt.show()
    x = torch.nn.functional.pad(x, (sr//10-1, 0))
    y = torch.nn.functional.conv1d(
        x[None,None], 
        f[None,None],
    )[0,0]
    return y
    # return convolve(x, f)

def fractional_comb_fir_multitap(x, f0, a, sr):
    if x.dim() == 1: # time
        x = x[None, None]
    elif x.dim() == 2: # channels x time
        x = x[None]

    assert x.dim() == 3 # batch x channels x time

    if not isinstance(f0, torch.Tensor):
        f0 = torch.tensor([[f0]], device=x.device, dtype=x.dtype)
    if f0.dim() == 0:
        f0 = f0[None, None]
    if not isinstance(a, torch.Tensor):
        a = torch.tensor([[a]], device=x.device, dtype=x.dtype)
    if a.dim() == 0:
        a = a[None, None]

    assert f0.dim() == 2 # out_channels x in_channels
    assert a.dim() == 2 # out_channels x in_channels

    l = sr/f0 # out_channels x in_channels
    t = torch.arange(sr//10-1, -1, step=-1, device=x.device) # kernel_size

    # Tensor method
    taps = torch.arange(1, 11, device=x.device, dtype=x.dtype)[..., None, None, None] # n_taps x 1 x 1 x 1
    delays = taps * l[None, ..., None] # n_taps x out_channels x in_channels x 1
    gains = a[None, ..., None] ** taps # n_taps x out_channels x in_channels x 1
    time = t[None, None, None] # 1 x 1 x 1 x kernel_size
    shifted_time = time - delays # n_taps x out_channels x in_channels x kernel_size
    f = (gains * torch.sinc(shifted_time)).sum(0) # out_channels x in_channels x kernel_size

    # Loop method
    # f = torch.zeros((f0.shape[0], f0.shape[1], sr//10), device=x.device) # out_channels x in_channels x kernel_size
    # for i in range(1, 11):
    #     delay = (i * l)[..., None] # out_channels x in_channels x 1
    #     gain = (a ** i)[..., None] # out_channels x in_channels x 1
    #     time = t[None, None] # 1 x 1 x kernel_size
    #     shifted_time = time - delay # out_channels x in_channels x kernel_size
    #     f += gain * torch.sinc(shifted_time)

    f[..., -1] = 1. # original signal (x[i])

    x = torch.nn.functional.pad(x, (sr//10-1, 0))

    y = torch.nn.functional.conv1d(
        x, # batch x in_channels x time
        f, # out_channels x in_channels x kernel_size
    ) # batch x out_channels x time

    return y

def fractional_comb_fir_multitap_sparse(x, f0, a, sr):
    if x.dim() == 1: # time
        x = x[None, None]
    elif x.dim() == 2: # channels x time
        x = x[None]

    assert x.dim() == 3 # batch x channels x time

    if not isinstance(f0, torch.Tensor):
        f0 = torch.tensor([[f0]], device=x.device, dtype=x.dtype)
    if f0.dim() == 0:
        f0 = f0[None, None]
    if not isinstance(a, torch.Tensor):
        a = torch.tensor([[a]], device=x.device, dtype=x.dtype)
    if a.dim() == 0:
        a = a[None, None]

    assert f0.dim() == 2 # out_channels x in_channels
    assert a.dim() == 2 # out_channels x in_channels

    l = sr/f0 # out_channels x in_channels
    t = torch.arange(sr//10-1, -1, step=-1, device=x.device) # kernel_size

    # Tensor method
    taps = torch.arange(1, 11, device=x.device, dtype=x.dtype)[..., None, None, None] # n_taps x 1 x 1 x 1
    delays = taps * l[None, ..., None] # n_taps x out_channels x in_channels x 1
    gains = a[None, ..., None] ** taps # n_taps x out_channels x in_channels x 1
    time = t[None, None, None] # 1 x 1 x 1 x kernel_size
    shifted_time = time - delays # n_taps x out_channels x in_channels x kernel_size
    f = (gains * torch.sinc(shifted_time)).sum(0) # out_channels x in_channels x kernel_size

    # Loop method
    # f = torch.zeros((f0.shape[0], f0.shape[1], sr//10), device=x.device) # out_channels x in_channels x kernel_size
    # for i in range(1, 11):
    #     delay = (i * l)[..., None] # out_channels x in_channels x 1
    #     gain = (a ** i)[..., None] # out_channels x in_channels x 1
    #     time = t[None, None] # 1 x 1 x kernel_size
    #     shifted_time = time - delay # out_channels x in_channels x kernel_size
    #     f += gain * torch.sinc(shifted_time)

    f[..., -1] = 1. # original signal (x[i])

    # Enforce sparsity
    f[abs(f)<1e-2] = 0.

    x = torch.nn.functional.pad(x, (sr//10-1, 0))

    y = torch.nn.functional.conv1d(
        x, # batch x in_channels x time
        f, # out_channels x in_channels x kernel_size
    ) # batch x out_channels x time

    return y

def single_fractional_comb_modulo(x, f0, a, sr):
    x = x.squeeze()
    l = sr/f0
    # l = (sr//f0)
    # import pdb; pdb.set_trace()
    t = torch.arange(0, sr//20)

    # f = t % l
    f = torch.remainder(t, l)
    a = (a ** (t/l))

    # f = (torch.sinc(f)) * a
    import pdb; pdb.set_trace()
    f = torch.sinc(f) * a
    # f = torch.sinc(f - f % 1.) * a

    # from matplotlib import pyplot as plt
    # plt.plot(f); plt.gcf().set_size_inches(10, 7.5); plt.show()

    # f = torch.flip(f, (0,))

    return convolve(x, f)

def single_comb_iir_fast(x, f0, a, sr):
    if x.dim() == 2:
        x = x.unsqueeze(0)
    assert x.dim() == 3
    f0 = float(f0)
    sr = int(sr)
    l = int(round(sr/f0))
    y = x.clone()
    step = l
    for i in range(l, x.shape[-1]-step, step):
            y[..., i:i+step] += a*y[..., i-l:i-l+step]
    return y

def single_fractional_comb_iir_faithful(x, f0, a, sr):
    if x.dim() == 2:
        x = x.unsqueeze(0)
    assert x.dim() == 3
    # Make the filter
    l = sr/f0
    t = (torch.arange( sr//10, device=x.device) - l).to(x.device)
    f = a * torch.sinc( t) #* torch.exp( -.1 * t**2)
    # f[0] = 1 # a[0] should be 1!
    f = torch.flip(f, (-1,))
    size = sr//10
    y = torch.zeros_like(x)
    y = torch.nn.functional.pad(y, (size-1, 0))
    for i in range(0, x.shape[-1]):
        # if i > 48:
        #     import pdb; pdb.set_trace()
        y[..., i+size-1] = (y[..., i:i+size] * f[None, None, :]).sum() + x[..., i]
    return y[..., size-1:]

# Take 2: Use an IIR comb filter, but apply using spectral division
# Much faster than above, and has a controlled DC bump, but a little hacky of course
def single_fractional_comb_fiir( x, f0, a, sr):
    # Make the filter
    l = sr/f0
    t = (torch.arange( sr//10, device=x.device) - l).to(x.device)
    f = -a * torch.sinc( t) * torch.exp( -.1 * t**2)
    f[0] = 1 # a[0] should be 1!

    l = x.shape[-1]
    # lp = ones( l//2+1).at[:int(l*500/sr)].set( linspace( 0, 1, int(l*500//sr))**2) # lowpass filter to remove pesky DC peak
    lp = torch.ones( l//2+1, device=x.device)
    lp[:80] = 0 # lowpass filter to remove pesky DC peak
    return torch.fft.irfft( lp * torch.fft.rfft( x, n=l) / torch.fft.rfft( f, n=l))[:x.shape[-1]] # change to fast conv instead?

def fractional_comb_fiir(x, f0, a, sr):
    if x.dim() == 2: # audio_channels x time
        x = x[None, None]
    if x.dim() == 3: # batch x audio_channels x time
        x = x[:, None]
    assert x.dim() == 4 # batch x feature_channels x audio_channels x time
    if not isinstance(f0, torch.Tensor):
        f0 = torch.tensor(f0).to(x.device)
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a).to(x.device)
    if f0.dim() == 0:
        f0 = f0[None, None]
    assert f0.dim() == 2
    # TODO something might be wrong here with the shape of `a`
    assert a.dim() == 0 or a.dim() == 2
    if a.dim() == 2:
        a = a[..., None]
    # Make the filter
    l = sr/f0
    t = (torch.arange(sr//10, device=x.device)[None, None, :] - l[..., None]).to(x.device)
    f = -a * torch.sinc(t) * torch.exp(-.1 * t**2)
    f[..., 0] = 1 # a[0] should be 1!

    l = x.shape[-1]
    # lp = ones( l//2+1).at[:int(l*500/sr)].set( linspace( 0, 1, int(l*500//sr))**2) # lowpass filter to remove pesky DC peak
    lp = torch.ones(l//2+1, device=x.device)
    lp[:80] = 0 # lowpass filter to remove pesky DC peak
    # from matplotlib import pyplot as plt
    # plt.plot(1/abs(torch.fft.rfft( f, n=l).squeeze())); plt.show()
    x_fft = torch.fft.rfft(x, n=l)
    f_fft = torch.fft.rfft(f, n=l)
    filtered_fft = (lp * x_fft / f_fft[None]).sum(2)
    return torch.fft.irfft(filtered_fft)[:x.shape[-1]] # change to fast conv instead?

def fractional_anticomb_interference_fiir(x, f0, a, sr, residual_mode=False):
    assert x.dim() == 3
    if not isinstance(f0, torch.Tensor):
        f0 = torch.tensor(f0).to(x.device)
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a).to(x.device)
    if f0.dim() == 0:
        f0 = f0[None, None]
    if a.dim() == 2:
        a = a[..., None]
    assert f0.dim() == 2
    # Make the filter
    l = sr/f0
    t = (torch.arange( sr//10, device=x.device)[None, None, :] - l[..., None]).to(x.device)
    f = -a * torch.sinc( t) * torch.exp( -.1 * t**2)
    f[..., 0] = 1 # a[0] should be 1!

    l = x.shape[-1]
    # lp = ones( l//2+1).at[:int(l*500/sr)].set( linspace( 0, 1, int(l*500//sr))**2) # lowpass filter to remove pesky DC peak
    lp = torch.ones( l//2+1, device=x.device)
    lp[:80] = 0 # lowpass filter to remove pesky DC peak
    filter = torch.fft.rfft(f, n=l)
    if residual_mode:
        filter = filter.cumprod(0)
        filter = torch.cat([torch.ones(1, filter.shape[1], filter.shape[2], device=filter.device), filter[:-1]], 0)
    else:
        idx = ~torch.eye(f0.shape[0], dtype=bool, device=filter.device)
        idx = idx.unsqueeze(2).repeat(1, 1, f0.shape[1])
        filter = filter.unsqueeze(0).repeat(f0.shape[0], 1, 1, 1)
        filter = filter[idx].view(f0.shape[0]-1, f0.shape[0], f0.shape[1], filter.shape[-1])
        filter = filter.prod(0)
    # filter = filter.prod(0, keepdims=True) / filter #TODO might cause gradient problems
    # scale to compensate for multiple filters
    if filter.shape[0] >= 3:
        filter = filter ** (1/(filter.shape[0]-1)) #TODO check if necessary
    return torch.fft.irfft(((lp * torch.fft.rfft( x, n=l))[:, None] * filter[None]))[:x.shape[-1]] # change to fast conv instead?

def fractional_anitcomb_fiir( x, f0, a, sr):
    # Make the filter
    if not isinstance(f0, torch.Tensor):
        f0 = torch.tensor(f0).to(x.device)
    if f0.dim() == 0:
        f0 = f0[None]
    l = sr/f0
    t = (torch.arange( sr//10, device=x.device)[:, None] - l[None, :]).to(x.device)
    f = -a * torch.sinc( t) * torch.exp( -.1 * t**2)
    f[0] = 1 # a[0] should be 1!
    # import pdb; pdb.set_trace()
    l = x.shape[-1]
    # lp = ones( l//2+1).at[:int(l*500/sr)].set( linspace( 0, 1, int(l*500//sr))**2) # lowpass filter to remove pesky DC peak
    lp = torch.ones( l//2+1, device=x.device)
    lp[:80] = 0 # lowpass filter to remove pesky DC peak
    # f /= abs(f).max()
    spectral_filter = torch.fft.rfft( f.T, n=l).prod(0)
    # from matplotlib import pyplot as plt
    # import pdb; pdb.set_trace()
    # spectral_filter /= abs(spectral_filter).max()
    return torch.fft.irfft( lp * torch.fft.rfft( x, n=l) * spectral_filter)[:x.shape[-1]] # change to fast conv instead?

def single_fractional_comb_anti_fiir( x, f0, a, sr):
    # Make the filter
    if not isinstance(f0, torch.Tensor):
        f0 = torch.tensor(f0).to(x.device)
    if f0.dim() == 0:
        f0 = f0[None]
    l = sr/f0
    t = (torch.arange( sr//10, device=x.device)[:, None] - l[None, :]).to(x.device)
    f = -a * torch.sinc( t) * torch.exp( -.1 * t**2)
    f[0] = 1 # a[0] should be 1!
    # import pdb; pdb.set_trace()
    l = x.shape[-1]
    # lp = ones( l//2+1).at[:int(l*500/sr)].set( linspace( 0, 1, int(l*500//sr))**2) # lowpass filter to remove pesky DC peak
    lp = torch.ones( l//2+1, device=x.device)
    lp[:80] = 0 # lowpass filter to remove pesky DC peak
    # f /= abs(f).max()
    spectral_filter = torch.fft.rfft( f.T, n=l).prod(0)
    # from matplotlib import pyplot as plt
    # import pdb; pdb.set_trace()
    # spectral_filter /= abs(spectral_filter).max()
    return torch.fft.irfft( lp * torch.fft.rfft( x, n=l) * spectral_filter)[:x.shape[-1]] # change to fast conv instead?


# Take 3: Use an explicit FIR comb filter
# f0 is intimately tied to a for loop, so this does not have a gradient
def single_fractional_comb_fir( x, f0, a, sr):
    n = 640 #int( 768*a)
    t = torch.linspace( 0, 2*torch.pi*n/sr, n+1)[:-1]

    # Take 1, straightforward take
    a = 0
    for f in torch.arange( f0, sr/2, f0):
        a += torch.cos( f*t)

    # Soften up the filter and use convolution
    a *= torch.hann_window( n)
    return convolve( x, a)




# Take 4: Use an FIR Dirichlet kernel comb filter
# Works fine, but is numerically unstable around k*pi even after adding a bunch of tricks
def single_fractional_comb_diric( x, f0, a, sr):
    T = 1025 #(640*a).astype( int)

    x = x.squeeze()

    # Vanilla Dirichlet kernel implementation (peaks galore!)
    def diric( x, N):
        x = x % (2*torch.pi)
        return torch.sin( N*x) / (N*torch.sin( x))

    # Replace k*pi areas with a polynomial approximation to avoid numerical instability
    def diric2( x, N):
        x = x % (2*torch.pi)
        th = .001
        c1 = (x<th)
        c2 = abs(x-torch.pi)<th
        c3 = (x > (2*torch.pi-th))
        c4 = ~(c1|c2|c3)
        return c1 * (1+(1-N**2)*x**2/6) + \
            c2 * (1+(1-N**2)*(x-torch.pi)**2/6) + \
            c3 * (1+(1-N**2)*(x-2*torch.pi)**2/6) + \
            c4 * torch.sin( N*x) / (N*torch.sin( x))

    # Auto-select n to not alias
    n = int( 2*((sr/f0)//2)-1)

    # r = torch.arange( 0, T*torch.pi, torch.pi).to(x.device)
    a = diric2( f0*(torch.arange( 0, T*torch.pi, torch.pi)+1e-3)/sr, n) * torch.kaiser_window( T, False, beta=16.)
    # a = diric2( f0*(torch.arange( 0, T*torch.pi, torch.pi)+1e-3)/sr, n)
    # a = diric( f0*(r+1e-3)/sr, n)
    # from matplotlib import pyplot as plt
    # plt.show()
    # plt.plot(a); plt.show()
    return convolve( x, a)


# Take 5: Use an overdriven sine as the filter kernel
# Cannot control the harmonics well, but is numerically stable
def single_fractional_comb_fir_od( x, f0, a, sr):
    n = 768
    t = torch.linspace( 0, 2*torch.pi*n/sr, n+1)[:-1]

    a = torch.tanh( 4*torch.cos( f0*t))

    a *= torch.hanning( n)
    return convolve( x, a, mode='same')