# import libraries
import torch
import numpy as np


def Cosine4AIF_ExtKety_deep(t, aif, ke, dt, ve, vp, device='cpu'):
    # offset time array
    t = t - aif[0] - dt
    ct = torch.zeros(np.shape(t)).to(device)
    shapes = t.size()
    ke = ke.repeat(1, shapes[1])
    ve = ve.repeat(1, shapes[1])
    vp = vp.repeat(1, shapes[1])
    aif4 = torch.zeros(np.shape(ke)).to(device)
    aif4.fill_(aif[4])
    aif2 = torch.zeros(np.shape(ke)).to(device)
    aif2.fill_(aif[2])

    cpBolus = (aif[1] * CosineBolus_deep(t, aif2, device=device))
    cpWashout = (aif[1] * (aif[3] * ConvBolusExp_deep(t, aif2, aif4, device=device)))
    ceBolus = (ke * (aif[1] * ConvBolusExp_deep(t, aif2, ke, device=device)))
    ceWashout = (ke * (aif[1]*aif[3] * ConvBolusExpExp_deep(t, aif2, aif4, ke, device=device)))

    cp = cpBolus + cpWashout
    ce = ceBolus + ceWashout

    ct[t > 0] = (vp[t > 0] * cp[t > 0]) + (ve[t > 0] * ce[t > 0])

    return ct


def Cosine4AIF_ExtKety_deep_aif(t, aif, ke, dt, ve, vp, device='cpu'):
    # offset time array
    t = t - aif[0, 0] - dt
    ct = torch.zeros(np.shape(t)).to(device)
    shapes = t.size()
    ke = ke.repeat(1, shapes[1])
    ve = ve.repeat(1, shapes[1])
    vp = vp.repeat(1, shapes[1])
    aif4 = torch.zeros(np.shape(ke)).to(device)
    aif4.fill_(aif[4, 0])
    aif2 = torch.zeros(np.shape(ke)).to(device)
    aif2.fill_(aif[2, 0])
    aif1 = aif[1].unsqueeze(1)
    aif3 = aif[3].unsqueeze(1)

    cpBolus = aif1 * CosineBolus_deep(t, aif2, device=device)
    cpWashout = aif1 * (aif3 * ConvBolusExp_deep(t, aif2, aif4, device=device))
    ceBolus = ke * (aif1 * ConvBolusExp_deep(t, aif2, ke, device=device))
    ceWashout = ke * (aif1 * (aif3 * ConvBolusExpExp_deep(t, aif2, aif4, ke, device=device)))

    cp = cpBolus + cpWashout
    ce = ceBolus + ceWashout

    ct[t > 0] = (vp[t > 0] * cp[t > 0]) + (ve[t > 0] * ce[t > 0])

    return ct


def Cosine8AIF_ExtKety_deep(t, aif, ke, dt, ve, vp, device='cpu'):
    # offset time array
    t = t - aif[0] - dt
    ct = torch.zeros(np.shape(t)).to(device)
    shapes = t.size()
    ke = ke.repeat(1, shapes[1])
    ve = ve.repeat(1, shapes[1])
    vp = vp.repeat(1, shapes[1])
    aif3 = torch.zeros(np.shape(ke)).to(device)
    aif3.fill_(aif[3])
    aif5 = torch.zeros(np.shape(ke)).to(device)
    aif5.fill_(aif[5])
    aif7 = torch.zeros(np.shape(ke)).to(device)
    aif7.fill_(aif[7])
    aif8 = torch.zeros(np.shape(ke)).to(device)
    aif8.fill_(aif[8])

    cpBolus = aif[2] * aif[7] * ConvBolusExp_deep(t, aif3, aif7, device=device)
    cpRecirc = aif[2] * aif[6] * CosineBolus_deep(t-aif[1], aif8, device=device)
    cpWashout = aif[2] * aif[4] * ConvBolusExp_deep(t-aif[1], aif8, aif5, device=device)

    ceBolus = ke * aif[2] * aif[7] * ConvBolusExpExp_deep(t, aif3, aif7, ke, device=device)
    ceRecirc = ke * aif[2] * aif[6] * ConvBolusExp_deep(t-aif[1], aif8, ke, device=device)
    ceWashout = ke * aif[2] * aif[4] * ConvBolusExpExp_deep(t-aif[1], aif8, aif5, ke, device=device)

    cp = cpBolus + cpRecirc + cpWashout
    ce = ceBolus + ceRecirc + ceWashout

    ct[t > 0] = vp[t > 0] * cp[t > 0] + ve[t > 0] * ce[t > 0]

    return ct


def Cosine8AIF_ExtKety_deep_aif(t, aif, ke, dt, ve, vp, device='cpu'):
    # offset time array
    t = t - aif[0, 0] - dt
    ct = torch.zeros(np.shape(t)).to(device)
    shapes = t.size()
    ke = ke.repeat(1, shapes[1])
    ve = ve.repeat(1, shapes[1])
    vp = vp.repeat(1, shapes[1])
    aif3 = torch.zeros(np.shape(ke)).to(device)
    aif3.fill_(aif[3, 0])
    aif4 = torch.zeros(np.shape(ke)).to(device)
    aif4.fill_(aif[4, 0])
    aif5 = torch.zeros(np.shape(ke)).to(device)
    aif5.fill_(aif[5, 0])
    aif6 = torch.zeros(np.shape(ke)).to(device)
    aif6.fill_(aif[6, 0])
    aif7 = torch.zeros(np.shape(ke)).to(device)
    aif7.fill_(aif[7, 0])
    aif8 = torch.zeros(np.shape(ke)).to(device)
    aif8.fill_(aif[8, 0])
    aif2 = aif[2].unsqueeze(1)

    cpBolus = aif2 * aif7 * ConvBolusExp_deep(t, aif3, aif7, device=device)
    cpRecirc = aif2 * aif6 * CosineBolus_deep(t-aif[1, 0], aif8, device=device)
    cpWashout = aif2 * aif4 * ConvBolusExp_deep(t-aif[1, 0], aif8, aif5, device=device)

    ceBolus = ke * aif2 * aif7 * ConvBolusExpExp_deep(t, aif3, aif7, ke, device=device)
    ceRecirc = ke * aif2 * aif6 * ConvBolusExp_deep(t-aif[1, 0], aif8, ke, device=device)
    ceWashout = ke * aif2 * aif4 * ConvBolusExpExp_deep(t-aif[1, 0], aif8, aif5, ke, device=device)

    cp = cpBolus + cpRecirc + cpWashout
    ce = ceBolus + ceRecirc + ceWashout

    ct[t > 0] = vp[t > 0] * cp[t > 0] + ve[t > 0] * ce[t > 0]

    return ct


def CosineBolus_deep(t, m, device='cpu'):
    z = m * t
    I = (z >= 0) & (z < (2 * np.pi))
    y = torch.zeros(np.shape(I)).to(device)
    y[I] = 1 - torch.cos(z[I])

    return y


def ConvBolusExp_deep(t, m, k, device='cpu'):
    tB = 2 * np.pi / m
    I1 = (t > 0) & (t < tB)
    I2 = t >= tB

    y = torch.zeros(np.shape(I1)).to(device)

    y[I1] = t[I1] * SpecialCosineExp_deep(k[I1]*t[I1], m[I1]*t[I1], device=device)
    y[I2] = tB[I2]*SpecialCosineExp_deep(k[I2]*tB[I2], m[I2]*tB[I2], device=device)*torch.exp(-k[I2]*(t[I2]-tB[I2]))

    return y


def ConvBolusExpExp_deep(t, m, k1, k2, device='cpu'):
    tol = 1e-4

    tT = tol / torch.abs(k2 - k1)

    Ig = (t > 0) & (t < tT)
    Ie = t >= tT
    y = torch.zeros(np.shape(t)).to(device)

    y[Ig] = ConvBolusGamma_deep(t[Ig], m[Ig], 0.5 * (k1[Ig] + k2[Ig]), device=device)
    y1 = ConvBolusExp_deep(t[Ie], m[Ie], k1[Ie], device=device)
    y2 = ConvBolusExp_deep(t[Ie], m[Ie], k2[Ie], device=device)
    y[Ie] = (y1 - y2) / (k2[Ie] - k1[Ie])

    return y


def ConvBolusGamma_deep(t, m, k, device='cpu'):
    tB = 2 * np.pi / m
    I1 = (t > 0) & (t < tB)
    I2 = t >= tB
    y = torch.zeros(np.shape(I1)).to(device)

    ce = SpecialCosineExp_deep(k * tB, m * tB, device=device)
    cg = SpecialCosineGamma_deep(k * tB, m * tB, device=device)

    y[I1] = (t[I1]**2) * SpecialCosineGamma_deep(k[I1] * t[I1], m[I1] * t[I1], device=device)
    y[I2] = tB[I2] * ((t[I2] - tB[I2]) * ce[I2] + tB[I2] * cg[I2]) * torch.exp(-k[I2] * (t[I2] - tB[I2]))

    return y


def SpecialCosineGamma_deep(x, y, device='cpu'):
    x2 = x**2
    y2 = y**2
    expTerm = (3+torch.div(y2, x2) * (1 - torch.exp(-x))) - (torch.div(y2 + x2, x) * torch.exp(-x))
    trigTerm = ((x2 - y2) * (1 - torch.cos(y))) - ((2 * x * y) * torch.sin(y))
    f = torch.div((trigTerm + (y2 * expTerm)), (y2 + x2)**2)

    return f


def SpecialCosineExp_deep(x, y, device='cpu'):
    expTerm = torch.div((1 - torch.exp(-x)), x)

    trigTerm = (x * (1 - torch.cos(y))) - (y * torch.sin(y))
    f = torch.div((trigTerm + (y**2 * expTerm)), (x**2 + y**2))

    return f
