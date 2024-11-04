import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import orbipy as op

import SM as sm

import event_integrator as evint
import model_integrator as modint

import warnings
warnings.filterwarnings(action='ignore', category=RuntimeWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)

#%%
def get_Phi(arr, model):
    '''
    Get Monodromy matrix
    arr  - array containing the trajectory of an orbit over a period
    model - an instance of the crtbp model
    
    Returns
    Monodromy matrix of the given orbit
    '''
    if model.polar:
        return arr[-25:].reshape(5, 5)
    else:
        return arr[-36:].reshape(6, 6)
    
#%%
def get_FM_T2(x0, z0, v0, n, stmmodel):
    '''
    Get Floquet multipliers by integrating an orbit only on half of a period
    x0, z0 - initial coordinates of an orbit
    v0 - initial velocity of an orbit
    n- number of times the orbit intersects xz-plane over half of a period
    stmmodel - an instance of the crtbp model, which computes the STM matrix

    Returns
    Floquet multipliers of an orbit
    '''
    A = np.eye(6)
    A[[1, 3, 5], [1, 3, 5]] *= -1

    Omega = np.array([[ 0, 1, 0],
                  [-1, 0, 0],
                  [ 0, 0, 0]])
    Ident = np.eye(3)

    Mat1 = np.zeros((6, 6))
    Mat1[:3, 3:] = -Ident
    Mat1[3:, :3] = Ident
    Mat1[3:, 3:] = -2 * Omega

    Mat2 = np.zeros((6, 6))
    Mat2[:3, :3] = -2 * Omega
    Mat2[:3, 3:] = Ident
    Mat2[3:, :3] = -Ident

    det4 = evint.event_detector(stmmodel, events=[op.eventY(count=n)])


    if stmmodel.polar:
        print('Cannot compute Monodromy matrix using half period State Transition Matrix')
    s = stmmodel.get_zero_state()
    s[[0, 2, 4]] = x0, z0, v0
    arr, _ = det4.prop(s, 0, 8*np.pi, last_state='last', ret_df=False)
    PhiHalf = get_Phi(arr[-1], stmmodel)
    Phi1 = A @ Mat1 @ PhiHalf.T @ Mat2 @ A @ PhiHalf
    return np.linalg.eigvals(Phi1)

#%%
def get_unityFM(cur, model, plane, n):
    '''
    Find Floquet Multiplier closest to unity
    
    cur - initial state vector of an orbit
    model - instance of the crtbp model 
    plane - 'y' or 'z': intersections with which plane are used in the event detector
    n - number of times an orbit intersects plane in one period

    Returns
    Orbit's Floquet Multiplier closest to unity
    '''
    if not model.stm:
        stmmodel = modint.crtbp3_model(model.const_set, polar=model.polar, stm=True)
    else:
        stmmodel = model

    stm_cur = stmmodel.get_zero_state()
    stm_cur[:6] = cur[:6]
    
    if plane == 'y':
        stm_det = evint.event_detector(stmmodel, events=[evint.eventY(count=n)])
        arr1, ev1 = stm_det.prop(stm_cur, 0., 10. * np.pi, ret_df=False, last_state='last')
        Phi1 = get_Phi(arr1[-1], stmmodel)

    elif plane == 'z':
        stm_det = evint.event_detector(stmmodel, events=[evint.eventY(terminal=False), evint.eventZ(count=n)])
        arr1, ev1 = stm_det.prop(stm_cur, 0., 10. * np.pi, ret_df=False, last_state='last')
        ev1y = ev1[ev1[:, 0] == 0]
        Phi1 = get_Phi(ev1y[-1], stmmodel)
        
    ind = np.argmin(abs(np.linalg.eig(Phi1)[0] - 1))
    mult = np.linalg.eig(Phi1)[0][ind]
    return mult


#%%
def unstable_direction(s, model, n=2, plane='y'):
    '''
    Calculates the vector of the unstable direction.

    Parameters:
    ----------
    s: state vector for which unstable direction is calculated
    model: CR3BP model
    n: number of points in the Poincare section of the orbit

    Returns:
    --------
    Vector corresponding to the unstable direction of the given state. This vector is 2-d if model preserves Jacobi
    constant, and is 3-d, otherwise.
    '''
    stmmodel = modint.crtbp3_model(model.const_set, polar=model.polar, stm=True)
    if plane == 'y':
        det = evint.event_detector(stmmodel, events=[op.eventY(count=n)])
    elif plane == 'z':
        det = evint.event_detector(stmmodel, events=[op.eventZ(count=n)])
    ystm = stmmodel.get_zero_state()
    ystm[:6] = s[:6]
    arr, ev = det.prop(ystm, 0., 20. * np.pi, ret_df=False)

    if model.polar:
        Phi = ev[-1, -25:].reshape(5, 5)
        eigvals, eigvecs = np.linalg.eig(Phi[0:3, 3:5].T @ Phi[0:3, 3:5])
        idx = 0 if eigvals[0].real > eigvals[1].real else 1
        d = eigvecs[:, idx].real
        if d[0] < 0.:
            d = -d
        d /= (d[0] ** 2 + d[1] ** 2) ** 0.5

    else:
        Phi = ev[-1, -36:].reshape(6, 6)
        eigvals, eigvecs = np.linalg.eig(Phi[0:3, 3:6].T @ Phi[0:3, 3:6])
        idx = 0 if eigvals[0].real > eigvals[1].real else 1
        d = eigvecs[:, idx].real
        if d[0] < 0.:
            d = -d
        d /= (d[0] ** 2 + d[1] ** 2 + d[2] ** 2) ** 0.5

    return d


#%%
def get_dv(s, e, goal_point, model, plane='y', n=1, verbose=False, dv0=1e-14, dx=1e-16):
    '''
    Get the magnitude of the initial state vector

    s - state vector
    e - direction of the unstable direction
    goal_point - goal poincare section point
    model - CR3BP model
    plane - 'y' or 'z': intersections with which plane are used in propagation
    n - the number of the plane crossings in the propagation of the orbit

    '''
    if plane == 'y':
        det = evint.event_detector(model, events=[op.eventY(count=n)])
    elif plane == 'z':
        det = evint.event_detector(model, events=[op.eventZ(count=n)])

    opt = sm.optimizer()
    opt.dx = dx
    opt.dxmax = 1
    opt.tolerance = 1e-20

    dv = dv0

    for j in range(50):
        if not opt.needNextStep(): break
        if model.polar:
            newconstants = np.zeros(4)
            newconstants[:3] = model.constants
            newconstants[-1] = model.jacobi(s)
            sr = model.vxvy2rphi(s[:6])
            # print(e)
            sr[3:] += dv * e
            s1 = model.get_zero_state()
            s1[:6] = model.rphi2vxvy(sr, newconstants)
        else:
            s1 = model.get_zero_state()
            s1[:6] = s[:6]
            s1[3:6] += dv * e
        
        if plane == 'y':
            s1[1] = 0
        elif plane == 'z':
            s1[2] = 0
        
        if s1[3]**2 + s1[4]**2 > 0:
            _, ev = det.prop(s1, 0., 100. * np.pi, ret_df=False, last_state='last')
                
        else:
            print('vx = vy = 0')
            model_ord = modint.crtbp3_model(model.const_set, stm=False, polar=False)
            det_ord = evint.event_detector(model_ord, events=[op.eventY(count=n)])
            s1_ord = model_ord.get_zero_state()
            s1_ord[[0, 2, 4]] = s1[[0, 2, 4]]
            _, ev = det_ord.prop(s1_ord, 0., 100. * np.pi, ret_df=False, last_state='last')
            if ev.shape[0] < n:
                print('s1_ord', n, s1_ord[0], s1_ord[2], s1_ord[4], sep=', ')
            
        dist = np.linalg.norm(ev[-1, 4:7] - goal_point) ** 2

        dv = opt.nextX(dv, dist)
    dv, dist = opt.getXY()
    if verbose:
        print('Minimum mistake is', dist, 'dv is', dv, 'j', j, 'dx', opt.getdx())
    return dv, dist


#%%
def correction(x, z, v, ev_ref, model, plane='y', corrK=18, corrL=0, ips=1, dx=1e-12, dv0=0.0,
               change_goal=True, verbose=False, draw=False, check=False, poincare=False, ret_dv=False):
    '''
    Integrate given state vector for corrK intersections with the y=0 plane using corrections.

    Parameters:
    ----------
    x, z, v - coordinates of the state vector
    ev_ref - goal Poincare section
    model - CR3BP model
    corrK - number of intersections with the y=0 plane the integration is completed for
    corrL - number of points in the Poincare section used in finding the size of correction dv at each step
    ips - number of intersections with the y=0 plane the state vector is propagated for at each step
    dx - optimisation parameter for finding dv
    dv0 - the initial guess of the dv
    verbose - if True, full print is delivered
    draw - if True, the output includes the plot of the resulting orbit projections
    check - if True, at each step of finding the dv after implementation of the correction,
            new state vector is propagated for corrL intersections with the y=0 plane
            and the resulting trajectory is plotted
    poincare - if True, the poincare section of the resulting orbit is returned
    ret_dv - if True, a list of corrections is returned

    Returns:
    --------
    The sum of the components vx^2 and vz^2 of the state vector at the t = T / 2 * l, l = 1, 2, ..., [corrK / n], where
    n - number of points in the Poincare section of the orbit.
    '''
    start = model.get_zero_state()
    start[[0, 2, 4]] = x, z, v
    if plane == 'y':
        det_corr = evint.event_detector(model, events=[op.eventY(count=ips)])
        arr_corr, ev_corr = det_corr.prop(start, 0., 20. * np.pi, last_state='last')

    elif plane == 'z':
        det_corr = evint.event_detector(model, events=[op.eventZ(count=ips)])
        arr_corr, ev_corr = det_corr.prop(start, 0., 20. * np.pi, last_state='last')

        det_newstart = evint.event_detector(model, events=[op.eventZ(count=1)])
        _, ev_newstart = det_newstart.prop(start, 0., 20. * np.pi, last_state='last')
        start = model.get_zero_state() 
        start[:6]= ev_newstart.iloc[-1, 4:10]
    
    n = ev_ref.shape[0]
    goal_mod = ev_ref.copy()
    dvs =[]

    if change_goal:
        goal_mod.iloc[-1, 4:10] = start[:6]
        goal_mod.iloc[:ips] = ev_corr
    if corrL == 0:
        corrL = int(1.5 * n)
    if poincare:
        section = [ev_corr]
    if draw:
        arrays = [arr_corr]
        ax = plotter.plot_proj(arr_corr, color='red');
    if verbose:
        print('i_corr', 0)
    #else:
    #    print('i_corr', 0, end=' ')

    # Целевая функция для поиска угла альфа
    d22 = 0
    d23 = 0
    sumdv = 0

    for i_corr in range(ips, corrK, ips):
        #print(i_corr)
        cur_corr = model.get_zero_state()
        cur_corr[:6] = ev_corr.iloc[-1, 4:10]
        dv = dv0
        e = unstable_direction(cur_corr, model=model, n=n, plane=plane)
        # Коррекция на corrL пересечений
        for j_corr in range(corrL):
            dv, dist = get_dv(cur_corr,
                              e,
                              goal_mod.iloc[((i_corr * 1 + j_corr) % n), 4:7],
                              model,
                              plane=plane,
                              n=j_corr + 1,
                              verbose=verbose,
                              dx=dx,
                              dv0=dv)
        if model.polar:
            newconstants = np.zeros(4)
            newconstants[:3] = model.constants
            newconstants[-1] = model.jacobi(cur_corr)
            sr = model.vxvy2rphi(cur_corr[:6])
            sr[3:] += dv * e
            cur_corr = model.get_zero_state()
            cur_corr[:6] = model.rphi2vxvy(sr, newconstants)
        else:
            cur_corr[3:6] += dv * e
        sumdv += np.linalg.norm(dv)
        # 'x', 'z', 'v', 'Point of correction implementation', 'dv', 'e0', 'e1'
        dvs.append([x, z, v, i_corr, dv, *e])

        # Проверка, что с коррекцией действительно можно пролететь corrL пересечений
        if check:
            if plane == 'y':
                det_info = evint.event_detector(model, events=[op.eventY(count=corrL)])
            elif plane == 'z':
                det_info = evint.event_detector(model, events=[op.eventZ(count=corrL)])
            arr_info, ev_info = det_info.prop(cur_corr, 0., 20. * np.pi, last_state='last')
            plotter.plot_proj(arr_info, color='r');
        # Пролет ips пересечений
        arr_corr, ev_corr = det_corr.prop(cur_corr, 0., 20. * np.pi, last_state='last')

        nCr = np.sum(ev_corr.t > 1e-4)
        ips1 = ips
        while nCr < ips:
            ips1 += 1
            if plane == 'y':
                det_extra = evint.event_detector(model, events=[op.eventY(count=ips1)])
            elif plane == 'z':
                det_extra = evint.event_detector(model, events=[op.eventZ(count=ips1)])
            arr_corr, ev_corr = det_extra.prop(cur_corr, 0., 100. * np.pi, last_state='last')
            nCr = np.sum(ev_corr.t > 1e-4)

        if change_goal and i_corr < n:
            goal_mod.iloc[i_corr:i_corr + ips] = ev_corr.iloc[-ips:]
            if i_corr + ips == n:
                goal_mod.iloc[-1, 4:10] = start[:6]
        if draw:
            plotter.plot_proj(arr_corr, ax=ax, color='rby'[i_corr % 3]);
            arrays.append(arr_corr)
        if poincare:
            section.append(ev_corr)
        if verbose:
            print(i_corr, f'({sumdv})')
        #else:
        #    print(i_corr, end=' ')

        # Вычисление целевой функции, когда пролетели половину периода
        if (i_corr + ips) % int(n / 2) == 0:
            d22 += abs(ev_corr.iloc[-1, 7]) ** 2 + abs(ev_corr.iloc[-1, 9]) ** 2
            # print('Computing d22')
            # print(f'i_corr {i_corr}, ips {ips}, n {n}')
            # print(f'vx {ev_corr.iloc[-1, 7]}')
            # print(f'vz {ev_corr.iloc[-1, 9]}')
            # print(f'd23 {d23}')
        # Вычисление целевой функции, когда пролетели период
        if (i_corr + ips) % n == 0:
            d23 += np.linalg.norm(start[:6] - ev_corr.iloc[-1, 4:10]) ** 2
            # print('Computing d23')
            # print(f'i_corr {i_corr}, ips {ips}, n {n}')
            # print(f'start {start[:6]}')
            # print(f'ev {ev_corr.iloc[-1, 4:10]}')
            # print(f'd23 {d23}')

    #print('d22', d22)
    #print('sum', sumdv)
    if verbose:
        print('sum', sumdv)
        print('d23', d23)
    if poincare and draw:
        return d22, pd.concat(section, ignore_index=True).iloc[:, 4:], pd.concat(arrays, ignore_index=True)
    if poincare:
        return d22, pd.concat(section, ignore_index=True).iloc[:, 4:]
    if ret_dv:
        if plane == 'y':
            return d22, dvs
        elif plane == 'z':
            return d23, dvs
    return d23
#%%

def find_v(cur, goal, model, plane='y', v_dx=1e-3, v_dxmax=1e-5, check_dx=False, change_goal=True, nes=4, tol=1e-12):
    '''
    For the fixed values of x1 and z1 find the initial velocity v1.

    Paramters:
    cur - current state vector, 6-dimensional array
    goal - Poincare section of the close periodic orbit
    plane - 'y' or 'z': the plane of Poincare section used
    v_dx - the size of the optimizer's first step
    v_dxmax - the maximum size of the optimizer's step
    check_dx - if True, print the difference between initial velocity v0 and new velocity v1
    change_goal - if True, starting from the n-th step of finding the velocity, instead of the Poincare section (goal)
                  of the reference orbit, Poincare section of the new orbit is used
    nes - number of extra steps of finding the velocity taken
    '''

    n = goal.shape[0]
    v1 = cur[4]
    goal_mod = goal.copy()
    if change_goal:
        goal_mod.iloc[n - 1, 4:] = cur

    for i in range(n + nes):
        opt = sm.optimizer()
        opt.dx = v_dx
        opt.dxmax = v_dxmax
        opt.tolerance = tol

        opt.dx /= 10 ** i
        opt.dxmax /= 10 ** i

        if change_goal and (i >= n) and (i < 2 * n - 1):
            if plane == 'y':
                det_i = evint.event_detector(model, events=[op.eventY(count=(i + 1) % n)])
            elif plane == 'z':
                det_i = evint.event_detector(model, events=[op.eventZ(count=(i + 1) % n)])
            _, ev1 = det_i.prop(cur, 0., 20. * np.pi, last_state='last')
            goal_mod[i % n] = ev1.iloc[-1]


        for j in range(100):
            if not opt.needNextStep(): break
            if plane == 'y':
                det_i = evint.event_detector(model, events=[op.eventY(count=i+1)])
            elif plane == 'z':
                det_i = evint.event_detector(model, events=[op.eventZ(count=i+1)])
            _, ev1 = det_i.prop(cur, 0., 20. * np.pi, last_state='last')

            dist = np.linalg.norm(ev1.iloc[-1, 4:10] - goal_mod.iloc[i % n, 4:10])
            v1 = opt.nextX(v1, dist)
            cur[4] = v1
            # if i == 0:
            #     print(f"i {i}\tj {j} evout shape {ev1.shape[0]}\nv1 {v1}\ndist {dist}\ngoal {goal_mod.iloc[i % n, 4], goal_mod.iloc[i % n, 6], goal_mod.iloc[i % n, 8]}\nev {ev1.iloc[-1, 4], ev1.iloc[-1, 6], ev1.iloc[-1, 8]}")
        v1, dist = opt.getXY()

        if check_dx:
            print(f"Step number {i+1}", end='\t')
            print(f"difference {abs(v1 - cur[4])}, \t number of optimization steps {j+1}, goal function {dist}")
        cur[4] = v1
    return v1

#%%
def find_v_T(cur, goal, model, plane='y', v_dx=1e-3, v_dxmax=1e-5, check_dx=False, change_goal=True, nes=4, tol=1e-12):
    n = goal.shape[0]
    T = np.array(goal.t.diff())
    T[0] = goal.t[0]
    T = np.resize(T, n+nes)

    v1 = cur[4]
    goal_mod = goal.copy()
    if change_goal:
        goal_mod.iloc[n - 1, 4:] = cur

    for i in range(n + nes):
        opt = sm.optimizer()
        opt.dx = v_dx
        opt.dxmax = v_dxmax
        opt.tolerance = tol

        opt.dx /= 10 ** i
        opt.dxmax /= 10 ** i

        if change_goal and (i >= n) and (i < 2 * n - 1):
            if plane == 'y':
                det_i = evint.event_detector(model, events=[op.eventY(count=(i + 1) % n)])
            elif plane == 'z':
                det_i = evint.event_detector(model, events=[op.eventZ(count=(i + 1) % n)])
            _, ev1 = det_i.prop(cur, 0., 20. * np.pi, last_state='last')
            goal_mod[i % n] = ev1.iloc[-1]
            T[i] = ev1.iloc[-1, 3] - T[:i].sum()

        for j in range(100):
            if not opt.needNextStep(): break
            ev1 = model.prop(cur, 0., T[:i + 1].sum())
            dist = np.linalg.norm(ev1.iloc[-1, 1:7] - goal_mod.iloc[i % n, 4:10])
            v1 = opt.nextX(v1, dist)
            cur[4] = v1           
        v1, dist = opt.getXY()
        if plane == 'y':
            det_i = evint.event_detector(model, events=[op.eventY(count=i + 1)])
        elif plane == 'z':
            det_i = evint.event_detector(model, events=[op.eventZ(count=i + 1)])
        _, ev1 = det_i.prop(cur, 0., 1000 * np.pi, last_state='last')
        T[i] = ev1.t[i] - T[:i].sum()


        if check_dx:
            print(f"Step number {i + 1}", end='\t')
            print(f"difference {abs(v1 - cur[4])}, \t number of optimization steps {j + 1}, goal function {dist}")
        cur[4] = v1
    return v1


#%%
def compute_goal_function(current_state, ev_ref, n, model, det, plane='y',
                          v_dx=1e-3, v_dxmax=1e-5, v_tol=1e-12, change_goal=True, nes=4,
                          corr=False, corrK=18, corrL=0, ips=1, ret_dv=False,
                          beta=0, verbose=False):
        '''
        For the fixed values of x1 and z1 find the initial velocity v1 and compute the goal function for finding alpha.

        Parameters:
        current_state - current state vector, 6-dimensional array
        ev_ref - Poincare section of the close periodic orbit
        n - integer, corresponding to the size of the reference orbit Poincare section
        model - cr3bp model
        det - event detector cirresponding to model
        plane - 'y' or 'z': the plane of Poincare section used
        v_dx - the size of the velocity optimiser first step
        v_dxmax - the maximum size of the velocity optimiser step
        v_tol - the tolerance of the velocity optimiser
        change_goal - if True, starting from the n-th step of finding the velocity, instead of the Poincare section (goal)
                    of the reference orbit, Poincare section of the new orbit is used
        nes - number of extra steps of finding the velocity taken
        corr - if True, corrections are applied
        corrK - the number of correction steps
        corrL - number of points in the Poincare section used at each correction step
        ips - number of intersections with the y=0 plane the state vector is propagated for at each correction step
        ret_dv - if True, a list of corrections is returned
        beta - if beta = 0,
           goal function is the distance between the initial state vector and the state vector propagated for one period;
           if beta = 1,
           goal function is the difference between the unity and the Floquet multiplier closest to the unity
        '''

        # Computation of the velocity
        v1 = find_v(current_state, ev_ref, model, plane=plane, v_dx=v_dx, v_dxmax=v_dxmax, tol=v_tol,
                    check_dx=False, change_goal=change_goal, nes=nes)

        # Computation of the goal functions
        # Computation of the distance between initial state vector and state vector propagated for one period

        print('First guess of the initial velocity', v1)
        dvs = 0
        if corr:
            x1, z1 = current_state[[0, 2]]
            if ret_dv:
                dist, dvs = correction(x1, z1, v1, ev_ref, model, plane=plane, corrK=corrK, corrL=corrL, ips=ips, change_goal=change_goal, ret_dv=ret_dv, verbose=True)
            else:
                dist = correction(x1, z1, v1, ev_ref, model, plane=plane, corrK=corrK, corrL=corrL, ips=ips, change_goal=change_goal, ret_dv=ret_dv)
        else:
            _, ev1 = det.prop(current_state, 0., 10. * np.pi, ret_df=False, last_state='last')
            if plane == 'y':
                dist = np.linalg.norm(ev1[-1, 4:7] - current_state[:3])
            elif plane == 'z':
                dist = np.linalg.norm(ev1[-1, 4:7] - ev1[0, 4:7])
        
        # Computation of Floquet multipliers
        mult = 1
        if beta > 0:
            mult = get_unityFM(current_state, model, plane, n)
        dist_mult = np.abs(mult - (1 + 0j))

        goal = dist_mult * beta + dist * (1 - beta)
        return goal, v1, mult, dist, dist_mult, dvs


#%%
def find_alpha_fm(ref, radius, alpha0, n, model, plane='y', goal=None,
              alpha_dx=1e-3, alpha_dxmax=1e-2, alpha_tol=1e-16, K=20,
              v_dx=1e-3, v_dxmax=1e-5, v_tol=1e-12, change_goal=True, nes=4,
              corr=False, corrK=18, corrL=0, ips=1, ret_dv=False,
              beta=0, verbose=False):
    '''
    For the fixed values x0, z0, v0, corresponding to the known periodical orbit, find the initial values x1, z1, v1
    of the next orbit of the family. It is supposed, that the distance between (x0, z0) and (x1, z1) is fixed and equals R.
    Then, x1 = x0 + R * cos(alpha), z1 = z0 + R * sin (alpha).

    Parameters:
    ref - reference periodic orbit state vector
    radius - the value of R
    alpha0 - first guess of alpha
    n - integer, corresponding to the size of the reference orbit Poincare section
    plane - 'y' or 'z': the plane of Poincare section used
    alpha_dx - the size of the angle optimiser first step
    alpha_dxmax - the maximum size of the angle optimiser step
    alpha_tol - the tolerance of the angle optimiser
    K - the number of angle optimiser steps
    v_dx - the size of the velocity optimiser first step
    v_dxmax - the maximum size of the velocity optimiser step
    v_tol - the tolerance of the velocity optimiser
    change_goal - if True, starting from the n-th step of finding the velocity, instead of the Poincare section (goal)
                  of the reference orbit, Poincare section of the new orbit is used
    nes - number of extra steps of finding the velocity taken
    corr - if True, corrections are applied
    corrK - the number of correction steps
    corrL - number of points in the Poincare section used at each correction step
    ips - number of intersections with the y=0 plane the state vector is propagated for at each correction step
    ret_dv - if True, a list of corrections is returned
    beta - if beta = 0,
           goal function is the distance between the initial state vector and the state vector propagated for one period;
           if beta = 1,
           goal function is the difference between the unity and the Floquet multiplier closest to the unity
    '''
           
    if plane == 'y':
        det = evint.event_detector(model, events=[evint.eventY(count=n)])
    elif plane == 'z':
        #det = evint.event_detector(model, events=[evint.eventY(terminal=False), evint.eventZ(count=n)])
        det = evint.event_detector(model, events=[evint.eventZ(count=n)])

    if goal is None:
        ref_g = model.get_zero_state()
        ref_g[[0, 2, 4]] = ref[[0, 2, 4]]
        _, goal = det.prop(ref_g, 0., 20. * np.pi, last_state='last')
    ev_ref = goal

    print('ref', ref[:6])
    opt0 = sm.optimizer()
    opt0.dx = alpha_dx
    opt0.dxmax = alpha_dxmax
    opt0.tolerance = alpha_tol
    opt0.output = True

    v1 = ref[4]
    distvals = np.array([100, v1, 0, 0, 0])

    for k in range(K):
        if not opt0.needNextStep(): break

        x1 = ref[0] + radius * np.cos(alpha0)
        z1 = ref[2] + radius * np.sin(alpha0)
        cur = model.get_zero_state()
        cur[0], cur[2], cur[4] = x1, z1, v1 
        #print("Angle", alpha0 / np.pi * 180)

        goal, v1, mult, dist, dist_mult, dvs = compute_goal_function(cur, ev_ref, n, model, det, plane=plane,
                                         v_dx=v_dx, v_dxmax=v_dxmax, v_tol=v_tol, change_goal=change_goal, nes=nes,
                                         corr=corr, corrK=corrK, corrL=corrL, ips=ips, ret_dv=ret_dv,
                                         beta=0, verbose=False)
        print(f"{k+1}/{K}\t Alpha {alpha0}\t Goal {goal}")
        if goal < distvals[0]:
            distvals[0] = goal
            distvals[1] = v1
            distvals[2] = mult
            distvals[3] = dist
            distvals[4] = dist_mult
            if ret_dv:
                dvsmin = dvs
        alpha0 = opt0.nextX(alpha0, goal**2)
    alpha0, goal = opt0.getXY()
    if ret_dv:
        return alpha0, distvals[1], distvals[3], dvsmin
    return alpha0, distvals[1], distvals[3], distvals[4]


#%%
def find_alpha(ref, radius, alpha0, n, model, plane='y', alpha_dx=1e-3, alpha_dxmax=1e-2, v_dx=1e-3, v_dxmax=1e-5, K=20,
               corr=False, corrK=18):
    '''
    For the fixed values x0, z0, v0, corresponding to the known periodical orbit, find the initial values x1, z1, v1
    of the next orbit of the family. It is supposed, that the distance between (x0, z0) and (x1, z1) is fixed and equals R.
    Then, x1 = x0 + R * cos(alpha), z1 = z0 + R * sin (alpha).

    Parameters:
    ref - reference periodic orbit state vector
    radius - the value of R
    alpha0 - first guess of alpha
    n - integer, corresponding to the size of the reference orbit Poincare section
    plane - 'y' or 'z': the plane of Poincare section used
    alpha_dx - the size of the optimizer's first step
    alpha_dxmax - the maximum size of the optimizer's step
    v_dx - the size of the velocity optimizer's first step
    v_dxmax - the maximum size of the velocity optimizer's step
    K - the number of angle optimizer steps
    corr - if True, corrections are applied
    corrK - the number of correction steps
    '''
    alpha0, v, d22, _ = find_alpha_fm(ref, radius, alpha0, n, model, plane=plane,
                                      alpha_dx=alpha_dx, alpha_dxmax=alpha_dxmax,
                                      v_dx=v_dx, v_dxmax=v_dxmax, K=K, corr=corr, corrK=corrK,
                                      beta=0)
    return alpha0, v, d22


#%%
def prop_corr(dv_list, model, draw=False, ax=None, lw=2):
    ips = int(dv_list[0, 3])

    start = model.get_zero_state()
    start[[0, 2, 4]] = dv_list[0, :3]

    det_corr = evint.event_detector(model, events=[op.eventY(count=ips)])
    arr_corr, ev_corr = det_corr.prop(start, 0., 20. * np.pi, last_state='last')
    arrays = [arr_corr]
    evs = [ev_corr]
    if draw:
        plotter = op.plotter.from_model(model, length_units='Mm')
        if ax is None:
            ax1 = plotter.plot_proj(arr_corr, lw=lw);
        else:
            plotter.plot_proj(arr_corr, ax=ax, lw=lw);
    for i in range(dv_list.shape[0]):
        cur_corr = model.get_zero_state()
        cur_corr[:6] = ev_corr.iloc[-1, 4:10]
        e = dv_list[i, 5:]
        dv = dv_list[i, 4]

        if model.polar:
            newconstants = np.zeros(4)
            newconstants[:3] = model.constants
            newconstants[-1] = model.jacobi(cur_corr)
            sr = model.vxvy2rphi(cur_corr[:6])
            sr[3:] += dv * e
            cur_corr = model.get_zero_state()
            cur_corr[:6] = model.rphi2vxvy(sr, newconstants)
        else:
            cur_corr[3:6] += dv * e

        # Пролет ips пересечений
        arr_corr, ev_corr = det_corr.prop(cur_corr, 0., 20. * np.pi, last_state='last')
        ncr = np.sum(ev_corr.t > 1e-4)
        ips1 = ips
        while ncr < ips:
            ips1 += 1
            det_extra = evint.event_detector(model, events=[op.eventY(count=ips1)])
            arr_corr, ev_corr = det_extra.prop(cur_corr, 0., 100. * np.pi, last_state='last')
            ncr = np.sum(ev_corr.t > 1e-4)

        arrays.append(arr_corr)
        evs.append(ev_corr.iloc[-ips:])
        if draw:

            if ax is None:
                plotter.plot_proj(arr_corr, ax=ax1, lw=lw);
            else:
                plotter.plot_proj(arr_corr, ax=ax, lw=lw);
    if draw and ax is None:
        return pd.concat(arrays, ignore_index=True), pd.concat(evs, ignore_index=True), ax1
    return pd.concat(arrays, ignore_index=True), pd.concat(evs, ignore_index=True)

#%%
def compute_goal_function_pickable(current_state, reference_state, n, plane='y',
                            v_dx=1e-3, v_dxmax=1e-5, v_tol=1e-12, change_goal=True, nes=4,
                            corr=False, corrK=18, corrL=0, ips=1, ret_dv=False,
                            beta=0, verbose=False, const_set='Earth-Moon (default)', stm=False, polar=False):
    '''
    Version of the function 'compute_goal_function', which does not take not pickable objects as arguments. Usually called from parallel function of joblib module.

    Parameters are the same as in the 'compute_goal_function'.

    Note to self: take as arguments the name of the model, stm and polar, then you can create the crtbp3_model not only for the Earth-Moon case.
    '''
    model = modint.crtbp3_model(const_set, stm=stm, polar=polar)

    if plane == 'y':
        det = evint.event_detector(model, events=[evint.eventY(count=n)])
    elif plane == 'z':
        det = evint.event_detector(model, events=[evint.eventY(terminal=False), evint.eventZ(count=n)])
    _, ev_ref = det.prop(reference_state, 0., 20. * np.pi, last_state='last')

    goal, v1, mult, dist, dist_mult, dvs = compute_goal_function(current_state, ev_ref, n, model, det, plane=plane,
                                                 v_dx=v_dx, v_dxmax=v_dxmax, v_tol=v_tol, change_goal=change_goal,
                                                 nes=nes,
                                                 corr=corr, corrK=corrK, corrL=corrL, ips=ips, ret_dv=ret_dv,
                                                 beta=beta, verbose=verbose)

    return [goal, v1, mult, dist, dist_mult, dvs]

#%%


def get_corr(x0, z0, dvvals):
    maskx0 = abs(dvvals[:, 0] - x0) < 1e-6 
    maskz0 = abs(dvvals[:, 1] - z0) < 1e-6
    mask = maskx0 & maskz0
    return dvvals[mask]


def get_corr_ind(ind, vals, dvvals):
    return get_corr(vals[ind, 0], vals[ind, 1], dvvals)


def prop_corr_fm(dv_list, model, draw=False, ax=None, lw=2):
    '''
    Get trajectory of an orbit with correction impulses

    dv_list: list with correction impulses
    model: CRTBP orbit
    draw: if True an orbit is plotted and axes are returned
    ax: axes for plotting
    lw: linewidth of the trajectory on the plot

    Returns
    Trajectory of the orbit, array with the xz-plane crossing points
    '''
    #print(dv_list)
    ips = int(dv_list[0, 3])

    start = model.get_zero_state()
    start[[0, 2, 4]] = dv_list[0, :3]
    
    if not model.stm:
        model.stm = True

    det_corr = evint.event_detector(model, events=[op.eventY(count=ips)])
    arr_corr, ev_corr = det_corr.prop(start, 0., 20. * np.pi, last_state='last', ret_df=False)
    arrays = [arr_corr]
    evs = [ev_corr]
    #plotter.plot_proj(arr_corr)
    
    Phi = get_Phi(arr_corr[-1], model)
        
    if draw:
        plotter = op.plotter.from_model(model, length_units='Mm')
        if ax is None:
            ax1 = plotter.plot_proj(arr_corr, lw=lw);
        else:
            plotter.plot_proj(arr_corr, ax=ax, lw=lw);
    
    for i in range(dv_list.shape[0]):
        cur_corr = model.get_zero_state()
        #print(ev_corr)
        cur_corr[:6] = ev_corr[-1, 4:10]
        e = dv_list[i, 5:]
        dv = dv_list[i, 4]
        
        if len(e) == 2:
            newconstants = np.zeros(4)
            newconstants[:3] = model.constants
            newconstants[-1] = model.jacobi(cur_corr)
            sr = model.vxvy2rphi(cur_corr[:6])
            sr[3:] += dv * e
            cur_corr = model.get_zero_state()
            cur_corr[:6] = model.rphi2vxvy(sr, newconstants)
        else:
            cur_corr[3:6] += dv * e
        

        # Пролет ips пересечений
        arr_corr, ev_corr = det_corr.prop(cur_corr, 0., 20. * np.pi, last_state='last', ret_df=False)
        ncr = np.sum(ev_corr[:, 3] > 1e-4)
        ips1 = ips
        while ncr < ips:
            ips1 += 1
            det_extra = evint.event_detector(model, events=[op.eventY(count=ips1)])
            arr_corr, ev_corr = det_extra.prop(cur_corr, 0., 100. * np.pi, last_state='last', ret_df=False)
            ncr = np.sum(ev_corr[:, 3] > 1e-4)

        Phi = np.dot(get_Phi(arr_corr[-1], model), Phi)
            
        arrays.append(arr_corr)
        evs.append(ev_corr[-ips:])
        if draw:
            if ax is None:
                plotter.plot_proj(arr_corr, ax=ax1, lw=lw);
            else:
                plotter.plot_proj(arr_corr, ax=ax, lw=lw);
    if draw and ax is None:
        return np.vstack(arrays), np.vstack(evs), Phi, ax1
    return np.vstack(arrays), np.vstack(evs), Phi