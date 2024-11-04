import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import orbipy as op

params={'rtol':1e-10, 'atol':1e-10, 'nsteps':100000, 'max_step':np.inf}
integrator = op.dopri5_integrator(params=params)
model = op.crtbp3_model('Earth-Moon (default)', integrator=integrator)

plotter = op.plotter.from_model(model, length_units='Mm')
scaler = op.scaler.from_model(model)
stmmodel = op.crtbp3_model('Earth-Moon (default)', stm=True)

def df2complex(data):
    if 'l6' in data.columns:
        fmNum = 6
    else: fmNum = 5
    for i in range(fmNum):
        data[f'l{i+1}'] = data[f'l{i+1}'].apply(lambda x: complex(x))

def trajectory2SI(orb):
    '''
    arr - trajectory in DU
    '''
    orb.x = scaler(orb.x, 'nd-km')
    orb.y = scaler(orb.y, 'nd-km')
    orb.z = scaler(orb.z, 'nd-km')
    
    orb.vx = scaler(orb.vx, 'nd/nd-km/s')
    orb.vy = scaler(orb.vy, 'nd/nd-km/s')
    orb.vz = scaler(orb.vz, 'nd/nd-km/s')

    orb.t = scaler(orb.t, 'nd-d')
    return orb


def generate_one_trajectory(s, t):
    '''
    s - initial state vector, dimensionless units,
    t - whole period, dimensionless unit
    
    returns arr, trajectory in SI
    '''
    arr1 = model.prop(s, 0, t/2);
    arr2 = model.prop(s, 0, -t/2);
    arr2.t = arr1.t.iloc[-1] - arr2.t
    arr = pd.concat([arr1, arr2[::-1]]).reset_index(drop=True)
    return trajectory2SI(arr)


def compute_properties(orb1, fam_name, saveFiles=False, planarLyapunov=False, index_start=0, request=False):
    '''
    Formates the columns in the order
   + x (км),z(км),v(км/с),t(дни), 
   + ax(км),ay(км),az(км),
   + dist_primary(км),dist_secondary(км),
   + dist_curve(км),
   + cj,
   + l1_r,l2_r,l3_r,l4_r,l5_r,l6_r,l1_im,l2_im,l3_im,l4_im,l5_im,l6_im,
   + stable (bool), stability_order(int)

    If request is True, prepares for sending like a request
    Else prepares for adding directly in the db

    orb - pandas DataFrame, which has state vectors in dimensionless units and and period in days
    fam_name - tag of the family
    
    '''
    orb = orb1.copy()
    orbColumns = orb.columns
    # Обработка множителей Флоке
    if 'l6_r' not in orbColumns:
        if 'l5_r' in orbColumns:
            orb['l6_r'] = [1] * orb.shape[0]
            orb['l6_im'] = [0] * orb.shape[0]
        elif 'l1' in orbColumns:
            if 'l6' not in orbColumns:
                orb['l6'] = [1 + 0j] * orb.shape[0]
                
            lr = pd.DataFrame(list([np.real(orb.l1), np.real(orb.l2), np.real(orb.l3), 
                                    np.real(orb.l4), np.real(orb.l5), np.real(orb.l6)]), 
                             index=['l1_r', 'l2_r', 'l3_r', 'l4_r', 'l5_r','l6_r']).T
            lim = pd.DataFrame(list([np.imag(orb.l1), np.imag(orb.l2), np.imag(orb.l3), 
                                     np.imag(orb.l4), np.imag(orb.l5), np.imag(orb.l6)]),
                              index=['l1_im', 'l2_im', 'l3_im', 'l4_im', 'l5_im', 'l6_im']).T
            orb = pd.concat([orb, lr, lim], axis=1)
            
    if 'unitCirclePairs' not in orbColumns:
        print('I have no information on stability, it is sad')
            
    # Расчет свойств орбит
    ax = []
    ay = []
    az = []
    distPrim = []
    distSec = []
    distCurve = [0]
    stable = []
    stabilityOrder = []
    arr_all = []
    coordinates = []
    
    cjNotIn = 'cj' not in orbColumns
    if cjNotIn: cj = []
    
    for i in range(orb.shape[0]):
        s = model.get_zero_state()
        s[[0, 2, 4]] = np.real(orb.iloc[i][['x', 'z', 'v']])
        t = np.real(scaler(orb.iloc[i]['t'], 'd-nd'))
        arr = generate_one_trajectory(s, t)
        if saveFiles:
            arr_all.append(arr.to_json(orient='records'))
            coordinates.append(arr.iloc[0, 1:])
#         plotter.plot_proj(arr)
        
        ax.append(arr.x.max() - arr.x.min())
        ay.append(arr.y.max() - arr.y.min())    
        az.append(arr.z.max() - arr.z.min())
        
        distPrim.append(min(np.linalg.norm(arr.iloc[:, 1:4] - [-scaler(model.mu, 'nd-km'), 0, 0], axis=1)))
        distSec.append(min(np.linalg.norm(arr.iloc[:, 1:4] - [scaler(model.mu1, 'nd-km'), 0, 0], axis=1)))
        
        if i > 0: 
            distCurve.append(distCurve[-1] + np.linalg.norm(s[[0, 2]] - orb.iloc[i-1][['x', 'z']]))
        if cjNotIn: 
            cj.append(model.jacobi(s))
        stable.append(True if orb.unitCirclePairs[i] == 2 else False)
        stabilityOrder.append(orb.unitCirclePairs[i])
    
    
    if planarLyapunov:
        orb['alpha'] = [0] * orb.shape[0]
        
        
    orb['ax'] = ax
    orb['ay'] = ay
    orb['az'] = az
    orb['dist_primary'] = distPrim
    orb['dist_secondary'] = distSec
    
    orb['dist_curve'] = distCurve
    orb.x = scaler(orb.x, 'nd-km')
    orb.z = scaler(orb.z, 'nd-km')
    orb.v = scaler(orb.v, 'nd/nd-km/s')
    
    if cjNotIn: orb['cj'] = cj
    
    orb['stable'] = stable
    orb['stability_order'] = stabilityOrder
    orb['t_period'] = orb['t']

    if request:    
        orb = orb[['x', 'z', 'v', 'alpha', 't_period', 
                'l1_r', 'l2_r', 'l3_r', 'l4_r', 'l5_r', 'l6_r', 
                'l1_im', 'l2_im', 'l3_im', 'l4_im', 'l5_im', 'l6_im', 
                'ax', 'ay', 'az', 
                'dist_primary', 'dist_secondary', 'dist_curve', 
                'cj', 'stable', 'stability_order']]
    else:
        orb['id'] = pd.DataFrame(range(index_start, index_start+orb.shape[0]))

        orb = orb[['id', 'x', 'z', 'v', 'alpha', 't', 
                'l1_r', 'l2_r', 'l3_r', 'l4_r', 'l5_r', 'l6_r', 
                'l1_im', 'l2_im', 'l3_im', 'l4_im', 'l5_im', 'l6_im', 
                'ax', 'ay', 'az', 
                'dist_primary', 'dist_secondary', 'dist_curve', 
                'cj', 'stable', 'stability_order']]
    
    if saveFiles:
        file_name = f'formatted files\\{fam_name}.csv'
        orb.to_csv(file_name, index=False)
    return orb


def compute_trajectories(orb, fam_name, saveFiles=False, index_start=0, request=False):
    '''
    orb - pandas DataFrame, which has state vectors in dimensionless units and and period in days
    '''
    arr_all=[]
    coordinates=[]
    
    for i in range(orb.shape[0]):
        s = model.get_zero_state()
        s[[0, 2, 4]] = np.real(orb.iloc[i][['x', 'z', 'v']])
        t = np.real(scaler(orb.iloc[i]['t'], 'd-nd'))
        arr = generate_one_trajectory(s, t)
        if request:
            orbit_id = pd.DataFrame([index_start + i] * arr.shape[0], columns=["orbit_id"])
            arr_all.append(pd.concat([orbit_id, arr], axis=1))
        else:
            arr_all.append(arr.to_json(orient='records'))
            coordinates.append([scaler(s[0], 'nd-km'),
                                scaler(s[1], 'nd-km'),
                                scaler(s[2], 'nd-km'), 
                                scaler(s[3], 'nd/nd-km/s'), 
                                scaler(s[4], 'nd/nd-km/s'),
                                scaler(s[5], 'nd/nd-km/s')])
            
    if request:
        arr_all = pd.concat(arr_all).reset_index(drop=True)
    
    else:
        orb_id = pd.DataFrame(range(index_start, index_start+orb.shape[0]), columns=['orbit_id'])
        arr_all = pd.DataFrame(np.array(arr_all, dtype=list), columns=['v'])

        coordinates = pd.DataFrame(coordinates, columns=['x', 'y', 'z', 'vx', 'vy', 'vz'])
        arr_all=pd.concat([orb_id, arr_all['v'], coordinates], axis=1)
    
    if saveFiles:
        file_name_traj = f'formatted files//{fam_name}_trajectories.csv'
        arr_all.to_csv(file_name_traj, index=False)

    return arr_all


def compute_poincare_sections(orb, fam_name, saveFiles=False, index_start=0, request=False):
    '''
    orb - pandas DataFrame, which has state vectors in dimensionless units and and period in days
    '''
    det = op.event_detector(model, 
                            events=[op.eventX(terminal=False), op.eventY(terminal=False), 
                                    op.eventZ(terminal=False), op.eventVX(terminal=False), 
                                    op.eventVY(terminal=False), op.eventVZ(terminal=False)])
    
    planes = ['x = 0', 'y = 0', 'z = 0', 'vx = 0', 'vy = 0', 'vz = 0']
    sectionList = []
    planeList = []
    orbit_id = []
    
    for i in range(orb.shape[0]):
        # print(i)
        s = model.get_zero_state()
        s[[0, 2, 4]] = orb[['x', 'z', 'v']].iloc[i]
        period = scaler(orb['t'].iloc[i], 'd-nd')
        _, ev = det.prop(s, 0, period+1e-10)
        
        ev['t'] = scaler(ev['t'], 'nd-d')
        ev['x'] = scaler(ev['x'], 'nd-km')
        ev['y'] = scaler(ev['y'], 'nd-km')
        ev['z'] = scaler(ev['z'], 'nd-km')
        ev['vx'] = scaler(ev['vx'], 'nd/nd-km/s')
        ev['vy'] = scaler(ev['vy'], 'nd/nd-km/s')
        ev['vz'] = scaler(ev['vz'], 'nd/nd-km/s')
        
        for j in range(6):
            section = ev[ev.e == j].copy()
            if len(section) > 0:
                section.iloc[:, j+1] = [0] * section.shape[0]
                plane = planes[j]
                if request:
                    section = section.iloc[:, 3:]
                    for k in range(section.shape[0]):
                        planeList.append(plane)
                        orbit_id.append(i + index_start)  
                else:
                    section = section.iloc[:, 3:].to_json(orient='records') 
                    planeList.append(plane)
                    orbit_id.append(i + index_start) 
                
                sectionList.append(section)
                
    planeList = pd.DataFrame(planeList, columns=["plane"])
    orbit_id = pd.DataFrame(orbit_id, columns=["orbit_id"])              
    
    if request:
        sectionList = pd.concat(sectionList).reset_index(drop=True)
        result = pd.concat([orbit_id, planeList, sectionList], axis=1)
    
    else:
        sectionList = pd.DataFrame(sectionList, columns=['v'])
        result = pd.concat([orbit_id, planeList, sectionList], axis=1)
    
    if saveFiles:
        file_name_traj = f'formatted files//{fam_name}_poincare_sections.csv'
        result.to_csv(file_name_traj, index=False)
    
    return result