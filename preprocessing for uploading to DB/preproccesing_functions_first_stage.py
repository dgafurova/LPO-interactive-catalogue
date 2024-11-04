import pandas as pd
import numpy as np
import orbipy as op

params={'rtol':1e-12, 'atol':1e-12, 'nsteps':100000, 'max_step':np.inf}
integrator = op.dopri5_integrator(params=params)
model = op.crtbp3_model('Earth-Moon (default)', integrator=integrator)

plotter = op.plotter.from_model(model, length_units='Mm')
scaler = op.scaler.from_model(model)
stmmodel = op.crtbp3_model('Earth-Moon (default)', stm=True)


def get_ind(data, num):
    indlist=[]
    ideal = np.linspace(data.min(), data.max(), num=num)
    for i in range(num):
        indlist.append((abs(data - ideal[i])).idxmin())
    return indlist


def get_stability_complex(orb, threshold=1e-2):
    # nums - число пар НЕ на единичной окружности
    nums = []
    if 'l6' in orb.columns:
        mlt = orb[['l1', 'l2', 'l3', 'l4', 'l5', 'l6']]
        fmNum = 6
    else:
        mlt = orb[['l1', 'l2', 'l3', 'l4', 'l5']]    
        fmNum = 5
        
    for i in range(orb.shape[0]):
        num = 0
        for j in range(fmNum):
            if abs(mlt.iloc[i, j]) - 1.0 > threshold :
                num += 1
        nums.append(num)
    nums = np.array(nums)
   
    stability = np.array(nums == 0, dtype=int)
    orb['stability'] = stability
    
    # число пар множителей на единичной окружности
    orb['unitCirclePairs'] = 2 - nums



def get_stability(orb, threshold=1e-2):
    
    if 'l6' in orb.columns:
        mlt = orb[['l1_r', 'l2_r', 'l3_r', 'l4_r', 'l5_r', 'l6_r',
                   'l1_im', 'l2_im', 'l3_im', 'l4_im', 'l5_im', 'l6_im']]
        numFM = 6
    else:
        mlt = orb[['l1_r', 'l2_r', 'l3_r', 'l4_r', 'l5_r',
                   'l1_im', 'l2_im', 'l3_im', 'l4_im', 'l5_im']]   
        numFM = 5

    # nums - число пар НЕ на единичной окружности
    nums = []
    for i in range(orb.shape[0]):
        num = 0
        for j in range(numFM):
            if abs(mlt.iloc[i, j] + mlt.iloc[i, numFM+j]*1j) - 1.0 > threshold:
                num += 1
        nums.append(num)
    
    nums = np.array(nums)
    stability = np.array(nums == 0, dtype=int)
    orb['stability'] = stability
    
    # число пар множителей на единичной окружности
    orb['unitCirclePairs'] = 2 - nums




def get_dists_planar(orb, LP):
    orb['dists'] = orb.x - LP


def get_dists(orb):
    dists = [0]
    for i in range(1, orb.shape[0]):
        dist = np.linalg.norm(orb[['x', 'z']].iloc[i] - orb[['x', 'z']].iloc[i - 1])
        dists.append(dists[-1] + dist)
    
    dists = pd.DataFrame(dists, columns=['dists'])
    orb['dists'] = dists


def get_cj(orb):
    cjs = []
    for i in range(orb.shape[0]):
        y = model.get_zero_state()
        y[[0, 2, 4]] = np.real(orb.iloc[i, :3])
        cj = model.jacobi(y)
        cjs.append(cj)
    orb['cj'] = cjs


def get_cj_and_periods(orb, nCr):
    det = op.event_detector(model, events=[op.eventY(count=nCr)])

    periods = []
    cjs = []
    for i in range(orb.shape[0]):
        y = model.get_zero_state()
        y[[0, 2, 4]] = np.real(orb.iloc[i, :3])
        _, evout = det.prop(y, 0., 20.*np.pi, ret_df=False, last_state='last')
        t = scaler(evout[-1, 3], 'nd-d')
        cj = model.jacobi(y)
        periods.append(t)
        cjs.append(cj)
    
    orb['t'] = periods
    orb['cj'] = cjs


