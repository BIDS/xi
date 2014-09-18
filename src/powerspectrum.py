"""
    powerspectrum and CIC
"""

__all__ = ["powerfromdelta", "cic"]

import numpy

try:
# lets try to use pyfftw
    import pyfftw
    irfftn = pyfftw.interfaces.numpy_fft.irfftn
    rfftn = pyfftw.interfaces.numpy_fft.rfftn
    fftshift = pyfftw.interfaces.numpy_fft.fftshift
    fftfreq = pyfftw.interfaces.numpy_fft.fftfreq
except ImportError:
    #maybe shall issue a warning here
    irfftn = numpy.fft.irfftn
    rfftn = numpy.fft.rfftn
    fftshift = numpy.fft.fftshift
    fftfreq = numpy.fft.fftfreq

def powerfromdelta(delta, boxsize, logscale=False, collapse_axes=None,
        deconvolve=False, cheat=None):
    """ delta is over density.
        this obviously does not correct for redshift evolution.
        returns a collapsed powerspectrum, 

        if collapse_axes is None, collapse all dimensions.
        if collapse_axes is not None, just collapse the selecte dimensions.
        if collapse_axes is [], do not collapse and return the full n-dim
        Pk. [in fft freq orderings, 0 freq is at 0, not at center ]

        returns k, P

        where k is a list of the bin centers.
           k[0] is the bins centers of the first uncollapsed dimension,
           k[-1] is the centers of the collapsed dimensions,
           when collapse is None, k is just the center of the collapsed
           dimensions.
        P is of shape (uncollapsed ... axis, len(k))
    
       The power spectrum is assumed to have the gadget convention,
       AKA, normalized to (2 * pi) ** -3 times sigma_8.

       if deconvolve is True, deconvolve the field in K space with
       the CIC kernel.
    """
    N = numpy.prod(delta.shape, dtype='f8')
    Ndim = len(delta.shape)
    BoxSize = numpy.empty(Ndim, dtype='f8')
    BoxSize[:] = boxsize

    # each dim has a different K0
    # (Dx)^3 = N / prod(BoxSize)
    # extra 2 * pi is from K0!
    K0 = 2 * numpy.pi / BoxSize

    if cheat is None:
        delta_k = rfftn(delta) / N
    else:
        delta_k = cheat

    if collapse_axes is None:
        collapse_axes = numpy.arange(len(delta_k.shape))

    full = numpy.array(delta_k.shape)
    half = numpy.array(delta_k.shape) // 2
    # last dim is already halved
    half[-1] = delta_k.shape[-1] - 1
    full[-1] = half[-1] * 2
    Kret = []

    for i in range(Ndim):
        kx = fftfreq(full[i]) * full[i]
        if i != Ndim - 1:
            if len(collapse_axes) != 0:
                kx = fftshift(kx)
        else:
            kx = kx[:delta_k.shape[i]]
        Kret.append(kx * K0[i])
    if len(collapse_axes) != 0:
        delta_k = fftshift(delta_k, axes=numpy.arange(len(delta.shape) - 1))

    if deconvolve:
        for dim, ki in enumerate(Kret):
            shape = numpy.ones(len(Kret), dtype='i4')
            shape[dim] = len(ki)
            ki = ki.reshape(shape)
            kernel = numpy.sinc(ki * (0.5 * BoxSize[i] / full[dim]) / numpy.pi) ** -2
            delta_k *= kernel

    P = numpy.abs(delta_k) ** 2 * K0.prod() ** -1 * Dplus ** 2
    if len(collapse_axes) == 0:
        return Kret, P
    else:
        return collapse(P, Kret, collapse_axes, logscale)


def collapse(field, ticks=None, axis=[], logscale=False):
    """ collapse axis of a field,
        tics are the coordinates of the axes.
        tics is of length field.shape
        and len(tics[i]) == field.shape[i]
        
        axis is a list of axis to collapse.
        the collapsed axis will be represented by one
        axis added to the end of the axes of the return
        value.
        returns

           tics, newfield
        where tics are the coordinates of the new field.
    """
    Ndim = len(field.shape)
    if ticks is None:
        ticks = [1.0 * numpy.arange(i) for i in field.shape]
    for i in range(Ndim):
      assert len(ticks[i]) == field.shape[i]

    if axis is None:
        axis = list(range(Ndim))

    if len(axis) == 0:
        return ticks, field

    axis = list(axis)

    preserve = []
    newticks = []
    dist = None
    binsize = 0
    for i in range(Ndim):
        if i in axis:
            if dist is None:
                dist = ticks[i] ** 2
                binsize = ticks[i].ptp() / field.shape[i]
            else:
                dist = dist[:, None] + ticks[i][None, :] ** 2
                binsize = max(binsize, ticks[i].ptp() / field.shape[i])
            dist.shape = -1
        else:
            preserve.append(i)
            newticks.append(ticks[i])

    dist **= 0.5
    dmin = 0
    dmax = dist.max()
    Nbins = dmax / binsize
    if not logscale:
        bins = numpy.linspace(dmin, dmax, Nbins, endpoint=True)
        center = 0.5 * (bins[1:] + bins[:-1])
    else:
        ldmin, ldmax = numpy.log10([dmin, dmax])
        bins = numpy.logspace(ldmin, ldmax, Nbins, endpoint=True)
        center = (bins[1:] * bins[:-1]) ** 0.5

    newticks.append(center)

    slabs = field.transpose(preserve + axis).reshape(-1, len(dist))

    dig = numpy.digitize(dist, bins)
#    suminv = 1.0 / numpy.bincount(dig, weights=dist, minlength=bins.size+1)[1:-1]
    suminv = 1.0 / numpy.bincount(dig, minlength=bins.size+1)[1:-1]
    newfield = numpy.empty((slabs.shape[0], len(center)))
    for i, slab in enumerate(slabs):
        kpk = slab #* dist
        kpksum = numpy.bincount(dig, weights=kpk, minlength=bins.size+1)[1:-1]
        newfield[i] = kpksum * suminv

    newfield.shape = [field.shape[i] for i in preserve] + [len(center)]

    return newticks, newfield


def cic(pos, Nmesh, boxsize, weights=1.0, dtype='f8'):
    """ CIC approximation from points to Nmesh,
        each point has a weight given by weights.
        This does not give density.
        pos is supposed to be row vectors. aka for 3d input
        pos.shape is (?, 3).

    """
    chunksize = 1024 * 16
    BoxSize = 1.0 * boxsize
    Ndim = pos.shape[-1]
    Np = pos.shape[0]
    dtype = numpy.dtype(dtype)
    mesh = numpy.zeros(shape=(Nmesh, ) * Ndim,
            dtype=dtype, order='C')
    flat = mesh.reshape(-1)
    neighbours = ((numpy.arange(2 ** Ndim)[:, None] >> \
            numpy.arange(Ndim)[None, :]) & 1)
    for start in range(0, Np, chunksize):
        chunk = slice(start, start+chunksize)
        if numpy.isscalar(weights):
          wchunk = weights
        else:
          wchunk = weights[chunk]
        gridpos = numpy.remainder(pos[chunk], BoxSize) * (Nmesh / BoxSize)
        intpos = numpy.intp(gridpos)
        for i, neighbour in enumerate(neighbours):
            neighbour = neighbour[None, :]
            targetpos = intpos + neighbour
            targetindex = numpy.ravel_multi_index(
                    targetpos.T, mesh.shape, mode='wrap')
            kernel = (1.0 - numpy.abs(gridpos - targetpos)).prod(axis=-1)
            add = wchunk * kernel
            u, label = numpy.unique(targetindex, return_inverse=True)
            flat[u] += numpy.bincount(label, add)
    return mesh

