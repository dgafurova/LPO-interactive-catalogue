# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 00:31:13 2018

@author: stasb

This module is intended for event detection technique.

Event defined as time when some function E(t,s) called event-function reaches
specific value E*. In other words event detection is a root-finding process
for EF(t,s)=E(t,s)-E* function.
Detection splits into two steps: root separation and root refinement.
Root separation is finding two consecutive times ti, tj and states si, sj
(calculated through integration) such as EF(ti,si)*EF(tj,sj) < 0.
Root refinement is calculation of time t* and state s* such as
|EF(t*,s*)| < eps, where eps is specified accuracy.

Root separation takes into account direction at which event-function reaches
its E* value:

direction <=> condition
    -1        EF(ti,si)>0 and EF(tj,sj)<0
     1        EF(ti,si)<0 and EF(tj,sj)>0
     0        EF(ti,si)*EF(tj,sj) < 0

Events can be terminal and non terminal. Terminal events tells integrator
when it should terminate integration process and therefore can be treated
as boundary conditions.


"""

import math
import pandas as pd
import numpy as np
from scipy.optimize import root as scipy_root
# from scipy.optimize import brentq
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d
from numba import njit, types  # compiler, types
import pkg_resources

from orbipy.solout import default_solout, event_solout
from model_integrator import base_model
from orbipy.plotter import plottable


class event_detector:
    '''
    Class event_detector is intended for event detection during
    integration of ODE system.
    '''
    columns = ['e', 'cnt', 'trm']

    def __init__(self, model, events, tol=1e-12, accurate_model=None):
        #        if not issubclass(model.__class__, base_model):
        if not isinstance(model, base_model):
            raise TypeError("model should be instance of base_model class or of a subclass thereof")
        self.model = model
        if accurate_model is not None and not isinstance(accurate_model, base_model):
            raise TypeError("accurate_model should be instance of base_model class or of a subclass thereof")
        self.accurate_model = model if accurate_model is None else accurate_model
        #        if not events:
        #            raise ValueError("Empty events list")
        self.events = events
        self.solout = event_solout(self.events)
        self.tol = tol

    def prop(self, s0, t0, t1,
             ret_df=True,
             last_state='none'):
        '''

        Parameters
        ----------

        last_state : str
            Make last state of trajectory consistent with 'last' terminal event
                - 'none' - do not change trajectory;
                - 'last' - 'last' event is terminal event with greater row index in event states DataFrame;
                - 'mint' - 'last' event is terminal event that occured earlier than other terminal events
        '''
        if self.model.polar:
            return self.prop_polar(s0, t0, t1, ret_df=ret_df, last_state=last_state)
        for e in self.events:
            if hasattr(e, 'reset'):
                e.reset()
        old_solout, old_solout2 = self.model.solout, self.accurate_model.solout
        self.model.solout = self.solout
        df = self.model.prop(s0, t0, t1, ret_df=False)
        evdf = self.accurate(ret_df=False)
        self.model.solout, self.accurate_model.solout = old_solout, old_solout2

        # make last state consistent with terminal events
        last_state = last_state.lower()
        if evdf.shape[0] > 0 and last_state != 'none':
            cev, arr = self.split_data(evdf)
            arr_trm = arr[cev[:, 2] == 1]
            if arr_trm.shape[0] > 0:
                if last_state == 'last':
                    df[-1] = arr_trm[-1]
                elif last_state == 'mint':
                    s = arr_trm[np.argmin(arr_trm[:, 0])]
                    df[-1] = s

        if ret_df:
            df = self.model.to_df(df)
            evdf = self.to_df(evdf)
        return df, evdf

    def accurate(self, ret_df=True):
        self.accurate_model.solout = default_solout()

        evout = []
        for e in self.solout.evout:
            event = self.events[e[0]]
            if event.accurate:
                ts = self.solout.out[e[2]]
                #                print('ts shape:', len(ts))
                t0 = ts[0]
                s0 = ts[1:]
                t1 = self.solout.out[e[2] + 1][0]
                #                print('root call:', 't0', t0, 't1', t1, 's0', s0)
                sa = self.root(event, t0, t1, s0)
            #                print('acc shape:', len(sa))
            else:
                sa = self.solout.out[e[2]]
            #                print('not acc shape:', len(sa))
            evout.append([e[0], e[1], event.terminal, *sa])

        if ret_df:
            df = self.to_df(evout)
            return df
        return np.array(evout)

    def root(self, event, t0, t1, s0):
        s_opt = [0]  # for saving calculated state during root calculation

        #        print('root finding', type(event))
        def froot(t, s0, t0):
            t = t[0]  # because scipy.root passes [t] instead of t
            if t == t0:
                s = np.array([t, *s0])
            else:
                s = self.accurate_model.prop(s0, t0, t, ret_df=False)[-1]
            s_opt[0] = s
            res = event(t, s[1:])
            #            print('t0', t0, 't', t, 's0', s0, 's', s, 'res', res)
            #            print(event, t, s[1:], '->', res)
            return res

        scipy_root(froot, args=(s0, t0), x0=t0, tol=self.tol)
        #        print('end of root finding')

        #       scipy.optimize.solve_ivp uses:
        #        brentq(froot, t0, t1, args=(s0, t0), xtol=self.tol, rtol=self.tol)
        return s_opt[0]

    def prop_polar(self, s0, t0, t1,
             ret_df=True,
             last_state='none'):
        '''

        Parameters
        ----------

        last_state : str
            Make last state of trajectory consistent with 'last' terminal event
                - 'none' - do not change trajectory;
                - 'last' - 'last' event is terminal event with greater row index in event states DataFrame;
                - 'mint' - 'last' event is terminal event that occured earlier than other terminal events
        '''
        n = self.model._constants.shape[0]
        new_constants = np.empty(n + 1)
        new_constants[:-1] = self.model._constants
        new_constants[-1] = self.model.jacobi(s0)

        for e in self.events:
            if hasattr(e, 'reset'):
                e.reset()
        old_solout, old_solout2 = self.model.solout, self.accurate_model.solout
        self.model.solout = self.solout
        df = self.model.prop_polar(s0, t0, t1, ret_df=False) #t, x, y, z, vx, vy, vz
        evdf = self.accurate_polar(new_constants, ret_df=False) # ev, cnt, trm, t, x, y, z, vx, vy, vz
        self.model.solout, self.accurate_model.solout = old_solout, old_solout2

        # make last state consistent with terminal events
        last_state = last_state.lower()
        if evdf.shape[0] > 0 and last_state != 'none':
            cev, arr = self.split_data(evdf)
            arr_trm = arr[cev[:, 2] == 1]
            if arr_trm.shape[0] > 0:
                if last_state == 'last':
                    df[-1] = arr_trm[-1]
                elif last_state == 'mint':
                    s = arr_trm[np.argmin(arr_trm[:, 0])]
                    df[-1] = s

        if ret_df:
            df = self.model.to_df(df)
            evdf = self.to_df(evdf)
        return df, evdf


    def accurate_polar(self, constants, ret_df=True):
        self.accurate_model.solout = default_solout()

        evout = []
        for e in self.solout.evout:
            event = self.events[e[0]]
            if event.accurate:
                ts = self.solout.out[e[2]]
                t0 = ts[0]
                s0 = ts[1:]
                t1 = self.solout.out[e[2] + 1][0]
                sa = self.root_polar(event, t0, t1, s0, constants)
            else:
                sa = self.solout.out[e[2]]
                sa.append(0)
                sa[1:] = self.model.rphi2vxvy(sa[1:], constants)
            evout.append([e[0], e[1], event.terminal, *sa])

        if ret_df:
            df = self.to_df(evout)
            return df
        return np.array(evout)

    def root_polar(self, event, t0, t1, s0, constants):
        s_opt = [0]  # for saving calculated state during root calculation

        def froot(t, s0, t0, constants):
            t = t[0]  # because scipy.root passes [t] instead of t
            if t == t0:
                s = np.array([t, *s0])
            else:
                s = self.accurate_model.prop_polar(s0, t0, t, ret_df=False, new_constants=constants)[-1]
            s_opt[0] = s
            res = event(t, s[1:])
            return res

        scipy_root(froot, args=(s0, t0, constants), x0=t0, tol=self.tol)
        return s_opt[0]

    def to_df(self, arr, columns=None):
        if columns is None:
            columns = event_detector.columns + self.model.columns
        #print(len(columns), arr.shape)
        return self.model.to_df(arr, columns=columns)  # , index_col)

    def split_data(self, data):
        '''
        Splits data by columns: ['e', 'cnt', 'trm'] and ['t', 'x', 'y' ,...]
        '''
        if isinstance(data, pd.DataFrame):
            #            d = data.reset_index()
            return data[event_detector.columns], data[self.model.columns]
        n = len(event_detector.columns)
        return data[:, :n], data[:, n:]


class base_event(plottable):
    '''
    Class base_event is a common interface for all event classes in OrbiPy.
    Event stores necessary data and calculates value of event-function which
    it represents. Event detection is a root finding of event-function.
    '''
    coord = 'e'

    def __init__(self,
                 value=0,
                 direction=0,
                 terminal=True,
                 accurate=True,
                 count=-1):
        self.value = value
        self.terminal = terminal
        self.direction = direction
        self.accurate = accurate
        self.count = count

    def __call__(self, t, s):
        return 0

    def get_df(self):
        return pd.DataFrame({self.coord: self.value}, index=[0])

    def plot_df(self, df, ax, projection, **kwargs):
        p = projection.split('-')
        c = self.coord
        kwargs['label'] = self.__repr__()
        if p[0] == c:
            ax.axvline(df[c].values, **kwargs)
        elif p[1] == c:
            ax.axhline(df[c].values, **kwargs)

    def __repr__(self):
        return self.__class__.__name__ + \
               ':[val]%r [dir]%r [trm]%r [acc]%r [cnt]%r' % (self.value,
                                                             self.direction, self.terminal, self.accurate, self.count)


class event_combine(base_event):
    '''
    Class event_combine combines list of events in one that looks
    like first event in list but acts like all of them simultaneously.
    This event occur when any of events in list occur.
    '''

    def __init__(self, events):
        if not events:
            raise ValueError('At least one event should be specified')
        self.events = events

    def __getattr__(self, attr):
        #        print('combine_getattr', attr)
        return getattr(self.events[0], attr)

    def __call__(self, t, s):
        ret = 1.0
        for e in self.events:
            ret *= e(t, s)
        return ret

    def __repr__(self):
        return 'event_combine: ' + self.events.__repr__()


class event_chain(base_event):
    '''
    Class event_chain looks like last event in chain but works like sequence
    of events: events should occur in specified order and only last event
    will behave like event.
    All events in chain should be terminal!
    '''

    def __init__(self, events, autoreset=False):
        if not events:
            raise ValueError('At least one event should be specified')
        self.events = events
        self.autoreset = autoreset
        self.last = len(self.events) - 1
        self.select_event(0)

    def select_event(self, idx):
        #        print('event_chain idx:', idx)
        self.idx = idx
        self.event_checker = event_solout([self.events[self.idx]])

    def reset(self):
        self.select_event(0)

    def __getattr__(self, attr):
        #        print('chain_getattr', attr)
        return getattr(self.events[self.last], attr)

    def __call__(self, t, s):
        if self.idx == self.last:
            ret = self.events[self.idx](t, s)
            if self.autoreset and \
                    self.event_checker(t, s) == -1:
                self.reset()
            return ret
        else:
            if self.event_checker(t, s) == -1:
                self.select_event(self.idx + 1)  # select next event
            # returning 0 will work because of strict inequalities in
            # event_solout
            return 0.0

    def __repr__(self):
        return 'event_chain: ' + self.events.__repr__()


class center_event(base_event):
    '''
    Class center_event is a base class for all events that uses
    center (point) in calculations.
    '''

    def __init__(self,
                 center=np.zeros(3),
                 value=0,
                 direction=0,
                 terminal=True,
                 accurate=True,
                 count=-1):
        super().__init__(value, direction, terminal, accurate, count)
        self.center = center

    def __repr__(self):
        return super().__repr__() + ' [center]%r' % self.center


class center_angle_event(center_event):
    '''
    Class center_angle_event is a base class for all events that uses
    center (point) and angle in calculations.
    '''

    def __init__(self,
                 center=np.zeros(3),
                 flip=False,
                 value=0,
                 direction=0,
                 terminal=True,
                 accurate=True,
                 count=-1):
        '''
        Parameters
        ----------

        value : float
            Angle that belongs to [-pi, pi] segment.

        flip : bool
            Flip angle symmetrical to Y-axis therefore
            break at -pi/pi will be to the right of Y-axis.
        '''
        super().__init__(center, value, direction, terminal, accurate, count)
        self.flip = flip

    def __repr__(self):
        return super().__repr__() + ' [left]%r' % self.left


class model_event(base_event):
    '''
    Class model_event is a base class for all events that uses
    model in calculations.
    '''

    def __init__(self,
                 model,
                 value=0,
                 direction=0,
                 terminal=True,
                 accurate=True,
                 count=-1):
        super().__init__(value, direction, terminal, accurate, count)
        self.model = model

    def __repr__(self):
        return super().__repr__() + ' [model]%r' % self.model


class eventT(base_event):
    coord = 't'

    def __call__(self, t, s):
        return t - self.value


class eventSinT(base_event):
    def __call__(self, t, s):
        return math.sin((t / self.value) * math.pi)


class eventX(base_event):
    coord = 'x'

    def __call__(self, t, s):
        return s[0] - self.value

    def to_code(self, i):
        return 'val%03d = x-(%.18f)' % (i, self.value)


class eventY(base_event):
    coord = 'y'

    def __call__(self, t, s):
        return s[1] - self.value

    def to_code(self, i):
        return 'val%03d = y-(%.18f)' % (i, self.value)


class eventZ(base_event):
    coord = 'z'

    def __call__(self, t, s):
        return s[2] - self.value

    def to_code(self, i):
        return 'val%03d = z-(%.18f)' % (i, self.value)


class eventVX(base_event):
    coord = 'vx'

    def __call__(self, t, s):
        return s[3] - self.value


class eventVY(base_event):
    coord = 'vy'

    def __call__(self, t, s):
        return s[4] - self.value


class eventVZ(base_event):
    coord = 'vz'

    def __call__(self, t, s):
        return s[5] - self.value


class eventAX(model_event):
    coord = 'ax'

    def __call__(self, t, s):
        return self.model.right_part(t, s, self.model.constants)[3] - self.value


class eventAY(model_event):
    coord = 'ay'

    def __call__(self, t, s):
        return self.model.right_part(t, s, self.model.constants)[4] - self.value


class eventAZ(model_event):
    coord = 'az'

    def __call__(self, t, s):
        return self.model.right_part(t, s, self.model.constants)[5] - self.value


class eventR(center_event):
    splits = 64

    def __call__(self, t, s):
        return ((s[0] - self.center[0]) ** 2 +
                (s[1] - self.center[1]) ** 2 +
                (s[2] - self.center[2]) ** 2) - self.value ** 2

    def get_df(self):
        alpha = np.linspace(0, 2 * np.pi, self.splits)
        c01 = self.center[0] + self.value * np.cos(alpha)
        c10 = self.center[1] + self.value * np.sin(alpha)
        c02 = self.center[0] + self.value * np.cos(alpha)
        c20 = self.center[2] + self.value * np.sin(alpha)
        c12 = self.center[1] + self.value * np.cos(alpha)
        c21 = self.center[2] + self.value * np.sin(alpha)
        z = np.zeros(self.splits, dtype=float)
        return pd.DataFrame({'x': np.hstack((c01, c02, z)),
                             'y': np.hstack((c10, z, c12)),
                             'z': np.hstack((z, c20, c21))})

    def plot_df(self, df, ax, projection, **kwargs):
        p = projection.split('-')
        all_prj = ('x-y', 'x-z', 'y-z')
        for i, prj in enumerate(all_prj):
            if projection in (prj, prj[::-1]):
                s = slice(i * self.splits, (i + 1) * self.splits)
                ax.plot(df[p[0]][s].values,
                        df[p[1]][s].values, **kwargs)
                break


class eventDR(center_event):
    def __call__(self, t, s):
        v = s[3:]
        r = s[:3] - self.center
        r = r / (r[0] ** 2 + r[1] ** 2 + r[2] ** 2) ** 0.5
        return r[0] * v[0] + r[1] * v[1] + r[2] * v[2]


class eventRdotV(center_event):
    def __call__(self, t, s):
        r = s[:3] - self.center
        v = s[3:6]
        return r[0] * v[0] + r[1] * v[1] + r[2] * v[2]

    def to_code(self, i):
        return '''val%03d = (x-%.18f)*vx+(y-%.18f)*vy+(z-%.18f)*vz''' % (i, *self.center)


class eventAlphaX(center_angle_event):
    def __call__(self, t, s):
        x, y = s[0] - self.center, s[1]
        if self.flip:
            x = -x
        angle = math.degrees(math.atan2(y, x))
        return angle - self.value


class eventOmegaX(center_event):
    def __call__(self, t, s):
        v = s[3:5]
        r = (s[0] - self.center, s[1])
        omega = (r[0] * v[1] - r[1] * v[0]) / (r[0] ** 2 + r[1] ** 2)
        return omega - self.value


class eventHyperboloidX(center_event):
    '''
    Circular hyperboloid oriented along X axis.
    XY-section is hyperbola defined by implicit equation:
        x^2/a^2-y^2/b^2 = 1
    '''
    param_t = 1.0
    splits = 64

    def __init__(self, a, b,
                 flip=False,
                 center=0.0,
                 value=0,
                 direction=0,
                 terminal=True,
                 accurate=True,
                 count=-1):
        super().__init__(center, 0, direction, terminal, accurate, count)
        self.flip = flip
        self.a = a
        self.b = b

    def __call__(self, t, s):
        x, y = s[0] - self.center, (s[1] ** 2 + s[2] ** 2) ** 0.5
        if self.flip:
            x = -x
        return x - self.a * (1 + (y / self.b) ** 2) ** 0.5

    def get_df(self):
        pt = np.linspace(-self.param_t,
                         self.param_t,
                         self.splits + 1)
        x = self.a * np.cosh(pt) + self.center
        if self.flip:
            x = -x
        y = self.b * np.sinh(pt)
        return pd.DataFrame({'x': x, 'y': y})

    def plot_df(self, df, ax, projection, **kwargs):
        #        p = projection.split('-')
        #        all_prj = ('x-y', 'x-z', 'y-z')
        all_prj = ('x-y', 'x-z')
        for i, prj in enumerate(all_prj):
            if projection in (prj,):
                ax.plot(df['x'].values,
                        df['y'].values, **kwargs)
                #            if projection in (prj[::-1],):
                #                ax.plot(df[p[1]].values,
                #                        df[p[0]].values, **kwargs)
                break


class eventParaboloidX(center_event):
    '''
    Circular paraboloid oriented along X axis.
    XY-section is parabola defined by explicit equation:
        x = y^2/a
    '''
    splits = 64

    def __init__(self, a,
                 flip=False,
                 center=0.0,
                 value=0,
                 direction=0,
                 terminal=True,
                 accurate=True,
                 count=-1):
        super().__init__(center, 0, direction, terminal, accurate, count)
        self.flip = flip
        self.a = a

    def __call__(self, t, s):
        x, y = s[0] - self.center, (s[1] ** 2 + s[2] ** 2) ** 0.5
        if self.flip:
            x = -x
        return x - y ** 2 / self.a


class eventConeX(center_angle_event):
    def __call__(self, t, s):
        x, y = s[0] - self.center, (s[1] ** 2 + s[2] ** 2) ** 0.5
        if self.flip:
            x = -x
        angle = math.degrees(math.atan2(y, x))
        return angle - self.value


class eventInsidePathXY(center_angle_event):
    '''
    For detection of event when trajectory crosses 3D surface made by
    revolution of plane path around X-axis.
    '''

    def __init__(self,
                 path,
                 center,
                 flip=False,
                 splits=720,
                 direction=0,
                 terminal=True,
                 accurate=True,
                 count=-1):
        super().__init__(center, flip, 0, direction, terminal, accurate, count)
        #        if len(center) < 2:
        #            raise TypeError('center should be iterable with 2 components (x, y)\n%r'%center)
        self.splits = splits
        self.set_path(path)

    def set_path(self, path):
        if not isinstance(path, np.ndarray) or path.ndim < 2:
            raise TypeError('path should be numpy array with 2 dimensions\n%r' % path)
        self.path = path
        x = self.path[:, 0] - self.center
        if self.flip:
            x = -x
        y = self.path[:, 1]
        theta = np.arctan2(y, x)
        order = np.argsort(theta)
        theta = theta[order]
        r = x ** 2 + y ** 2
        r = r[order]
        #        self.rint = interp1d(theta[:-1], r[:-1], fill_value='extrapolate', kind='cubic')
        self.rint = InterpolatedUnivariateSpline(theta[:-1], r[:-1])
        # theta is uniformly distributed strictly increasing array
        self.theta = np.linspace(-np.pi, np.pi, self.splits)
        self.r = self.rint(self.theta)

    def theta_r(self, t, s):
        x, y, z = s[0] - self.center, s[1], s[2]
        if self.flip:
            x = -x
        r = x ** 2 + y ** 2 + z ** 2

        #        if y < 0.0:
        #            theta = math.atan2(-(y**2+z**2)**0.5, x)
        #        else:
        theta = math.atan2((y ** 2 + z ** 2) ** 0.5, x)
        return theta, r

    def interp(t, s, center, flip, x_arr, y_arr):
        '''
        Fast linear interpolation routine.
        x_arr should be uniformly distributed and sorted.
        '''
        # calculate r, theta
        x, y, z = s[0] - center, s[1], s[2]
        if flip:
            x = -x

        r = x ** 2 + y ** 2 + z ** 2

        #        if y < 0.0:
        #            theta = np.arctan2(-(y**2+z**2)**0.5, x)
        #        else:
        theta = math.atan2((y ** 2 + z ** 2) ** 0.5, x)

        # linear interpolation part
        i = types.int64((theta - x_arr[0]) // (x_arr[1] - x_arr[0]))
        #        i = math.floor((theta-x_arr[0]) / (x_arr[1]-x_arr[0]))
        r1 = y_arr[i] + (y_arr[i + 1] - y_arr[i]) / (x_arr[i + 1] - x_arr[i]) * (theta - x_arr[i])
        #        print(t, s, r, r1, r-r1)
        return (r - r1)

    interp = \
        njit(cache=True)(interp).compile("f8(f8,f8[:],f8,b1,f8[:],f8[:])")

    # interp = \
    # compiler.compile_isolated(interp,
    #                           [types.double, types.double[:], types.double,
    #                            types.boolean, types.double[:], types.double[:]],
    #                           return_type=types.double).entry_point

    def __call__(self, t, s):
        return eventInsidePathXY.interp(t, s, self.center, self.flip,
                                        self.theta, self.r)

    def deprecated__call__(self, t, s):
        theta, r = self.theta_r(t, s)
        r1 = self.rint(theta)
        print(r, r1)
        return r - r1  # inside path (event < 0)

    def get_df(self):
        return pd.DataFrame({'x': self.path[:, 0],
                             'y': self.path[:, 1]})

    def plot_df(self, df, ax, projection, **kwargs):
        p = projection.split('-')
        if projection in ('x-y', 'y-x'):
            ax.plot(df[p[0]].values,
                    df[p[1]].values, **kwargs)


class eventSplitLyapunov(center_angle_event):
    '''
    For detection of event when trajectory crosses 3D surface made by
    revolution of plane Lyapunov orbit around X-axis.
    '''

    def __init__(self,
                 lyapunov_orbit_half,
                 center,
                 flip=False,
                 #                 fname=pkg_resources.resource_filename(__name__, 'data/hlyapunov_sel1.csv'),
                 #                 orbit_idx=-500,
                 split_theta=1.8461392981282345,
                 left=True,
                 splits=720,
                 direction=0,
                 terminal=True,
                 accurate=True,
                 count=-1):
        super().__init__(center, flip, 0, direction, terminal, accurate, count)
        self.splits = splits
        self.split_theta = split_theta
        self.left = left
        self.set_orbit(lyapunov_orbit_half)

    def set_orbit(self, orbit):
        if not isinstance(orbit, np.ndarray) or orbit.ndim < 2:
            raise TypeError('orbit should be numpy array with 2 dimensions\n%r' % orbit)
        # orbit should be np.array, x->orbit[:,0], y->orbit[:,1]
        self.orbit = orbit.copy()
        mid_idx = self.orbit.shape[0] // 2
        if self.orbit[mid_idx, 1] < 0:  # flip orbit
            self.orbit[:, 1] = -self.orbit[:, 1]
        x = self.orbit[:, 0] - self.center
        if self.flip:
            x = -x
        y = self.orbit[:, 1]
        theta = np.arctan2(y, x)
        #        theta[theta < 0] += np.pi
        order = np.argsort(theta)
        theta = theta[order]
        r = x ** 2 + y ** 2
        r = r[order]
        self.rint = InterpolatedUnivariateSpline(theta, r)
        # theta is uniformly distributed strictly increasing array
        self.theta = np.linspace(0.0, np.pi, self.splits)
        self.r = self.rint(self.theta)

    def theta_r(self, t, s):
        x, y, z = s[0] - self.center, s[1], s[2]
        if self.flip:
            x = -x
        r = x ** 2 + y ** 2 + z ** 2

        theta = math.atan2((y ** 2 + z ** 2) ** 0.5, x)

        return theta, r

    def get_xy(self, theta):
        r = self.rint(theta) ** 0.5
        x = r * math.cos(theta)
        if self.flip:
            x = -x
        x += self.center
        y = r * math.sin(theta)
        return np.array([x, y])

    def interp(t, s, center, flip, x_arr, y_arr, theta_s, left):
        '''
        Fast linear interpolation routine.
        x_arr should be uniformly distributed and sorted.
        '''
        # calculate r, theta
        x, y, z = s[0] - center, s[1], s[2]
        if flip:
            x = -x

        theta_xy = math.atan2(math.fabs(y), x)
        #        print(theta_xy)
        #        print('theta_xy, theta_s', theta_xy, theta_s)
        if (not left and (theta_xy > theta_s)) or (left and (theta_xy < theta_s)):
            return -1.0

        theta = math.atan2((y ** 2 + z ** 2) ** 0.5, x)

        r = x ** 2 + y ** 2 + z ** 2

        # linear interpolation part
        i = types.int64((theta - x_arr[0]) // (x_arr[1] - x_arr[0]))
        #        i = math.floor((theta-x_arr[0]) / (x_arr[1]-x_arr[0]))
        r1 = y_arr[i] + (y_arr[i + 1] - y_arr[i]) / (x_arr[i + 1] - x_arr[i]) * (theta - x_arr[i])
        #        print(t, s, r, r1, r-r1)
        return (r - r1)

    interp = \
        njit(cache=True)(interp).compile("f8(f8,f8[:],f8,b1,f8[:],f8[:],f8,b1)")

    # interp = \
    # compiler.compile_isolated(interp,
    #                           [types.double, types.double[:], types.double,
    #                            types.boolean, types.double[:], types.double[:],
    #                            types.double, types.boolean],
    #                           return_type=types.double).entry_point

    def __call__(self, t, s):
        return eventSplitLyapunov.interp(t, s, self.center, self.flip,
                                         self.theta, self.r, self.split_theta,
                                         self.left)

    def deprecated__call__(self, t, s):
        theta, r = self.theta_r(t, s)
        theta_xy = math.atan2(math.fabs(s[1]), s[0])
        if (not self.left and (theta_xy > self.split_theta)) or (self.left and (theta_xy < self.split_theta)):
            return -1
        r1 = self.rint(theta)
        print(r, r1)
        return r - r1  # inside path (event < 0)

    def get_df(self):
        return pd.DataFrame({'x': self.orbit[:, 0],
                             'y': self.orbit[:, 1]})

    def plot_df(self, df, ax, projection, **kwargs):
        p = projection.split('-')
        if projection in ('x-y', 'y-x'):
            ax.plot(df[p[0]].values,
                    df[p[1]].values, **kwargs)


class eventSPL(model_event):
    _data = {}
    _compiled_find_index = None
    _compiled_interp = None

    def __init__(self,
                 model,
                 jc,
                 point='L1',
                 left=True,
                 #                 value=0,
                 direction=0,
                 terminal=True,
                 accurate=True,
                 count=-1):
        super().__init__(model, 0, direction, terminal, accurate, count)
        self.point = point
        self.left = left
        self.flip = (point == 'L1')
        self.center = model.__getattribute__(point)
        if eventSPL._compiled_find_index is None:
            eventSPL._compiled_find_index = \
                njit(cache=True)(eventSPL.find_index).compile("i8(f8[:],f8)")
            # eventSPL._compiled_find_index = \
            #     compiler.compile_isolated(eventSPL.find_index,
            #                               [types.double[:], types.double],
            #                               return_type=types.int64).entry_point
        if eventSPL._compiled_interp is None:
            eventSPL._compiled_interp = \
                njit(cache=True)(eventSPL.interp).compile("f8(f8,f8[:],f8,b1,f8[:],f8[:],f8,b1)")

            # eventSPL._compiled_interp = \
            #     compiler.compile_isolated(eventSPL.interp,
            #                   [types.double, types.double[:], types.double,
            #                    types.boolean, types.double[:], types.double[:],
            #                    types.double, types.boolean],
            #                   return_type=types.double).entry_point
        self.load_data(jc)

    def find_index(arr, val):
        '''
        Find index of first element lower than val
        '''
        for i in range(len(arr)):
            if arr[i] < val:
                return i
        return -1

    def find_index_old(arr, val):
        mask = arr <= val
        return np.argmax(mask)

    def set_jc(self, jc):
        sp = eventSPL._data[self.point]['SP']
        pr = eventSPL._data[self.point]['PR']
        idx = eventSPL._compiled_find_index(sp[:, 1], jc)
        if idx < 0:
            raise RuntimeError("Unable to find suitable orbit for Jacobi constant %f" % self.jc)
        self.split_theta = sp[idx, 2]
        self.selected_jc = sp[idx, 1]
        self.r = pr[idx].copy()
        self.jc = jc

    def load_data(self, jc):
        '''
        HLY = Horizontal Lyapunov Orbit Family
        PR = polar representation for HLY
        SP = Split Point
        '''
        if self.point in eventSPL._data:
            pr = eventSPL._data[self.point]['PR']
            sp = eventSPL._data[self.point]['SP']
        else:
            fname = 'HLY_' + self.point + '_' + self.model.const_set
            sp_fname = 'SP_' + fname + '.csv'
            pr_fname = 'PR_' + fname + '.npy'
            sp_path = pkg_resources.resource_filename(__name__, 'data/families/' + sp_fname)
            pr_path = pkg_resources.resource_filename(__name__, 'data/families/' + pr_fname)
            sp = np.loadtxt(sp_path)
            pr = np.load(pr_path)
            eventSPL._data[self.point] = {'SP': sp, 'PR': pr}

        self.set_jc(jc)
        self.theta = np.linspace(0., np.pi, 720)
        self.rint = InterpolatedUnivariateSpline(self.theta, self.r)

    def theta_r(self, t, s):
        x, y, z = s[0] - self.center, s[1], s[2]
        if self.flip:
            x = -x
        r = x ** 2 + y ** 2 + z ** 2

        theta = math.atan2((y ** 2 + z ** 2) ** 0.5, x)

        return theta, r

    def get_xy(self, theta):
        r = self.rint(theta) ** 0.5
        x = r * math.cos(theta)
        if self.flip:
            x = -x
        x += self.center
        y = r * math.sin(theta)
        return np.array([x, y])

    def interp(t, s, center, flip, x_arr, y_arr, theta_s, left):
        '''
        Fast linear interpolation routine.
        x_arr should be uniformly distributed and sorted.
        '''
        # calculate r, theta
        x, y, z = s[0] - center, s[1], s[2]
        if flip:
            x = -x

        theta_xy = math.atan2(math.fabs(y), x)
        #        print(theta_xy)
        #        print('theta_xy, theta_s', theta_xy, theta_s)
        if (not left and (theta_xy > theta_s)) or (left and (theta_xy < theta_s)):
            return -1.0

        theta = math.atan2((y ** 2 + z ** 2) ** 0.5, x)

        r = x ** 2 + y ** 2 + z ** 2

        # linear interpolation part
        i = types.int64((theta - x_arr[0]) // (x_arr[1] - x_arr[0]))
        #        i = math.floor((theta-x_arr[0]) / (x_arr[1]-x_arr[0]))
        r1 = y_arr[i] + (y_arr[i + 1] - y_arr[i]) / (x_arr[i + 1] - x_arr[i]) * (theta - x_arr[i])
        #        print(t, s, r, r1, r-r1)
        return (r - r1)

    def __call__(self, t, s):
        return eventSPL._compiled_interp(t, s, self.center, self.flip,
                                         self.theta, self.r, self.split_theta,
                                         self.left)


class eventFOV(center_event):
    def __init__(self,
                 orbit,
                 center,
                 r,
                 direction=0,
                 terminal=True,
                 accurate=True,
                 count=-1):
        super().__init__(center, r, direction, terminal, accurate, count)
        self.set_orbit(orbit)

    def set_orbit(self, orbit):
        if not isinstance(orbit, np.ndarray) or orbit.ndim < 2 or orbit.shape[1] < 4:
            raise TypeError('[orbit] should be numpy array of (n, 4) shape: (t,x,y,z)\n%r' % orbit)
        self.orbit = orbit
        self.oint = interp1d(self.orbit[:, 0],
                             self.orbit[:, 1:4],
                             axis=0,
                             kind='cubic',
                             fill_value='extrapolate')

    def __call__(self, t, s):
        cone = self.oint(t)[0]
        cone_c = self.center - cone
        cone_s = s[:3] - cone
        #        print(cone_c, cone_s)
        #        try:
        d_c = (cone_c[0] ** 2 + cone_c[1] ** 2 + cone_c[2] ** 2) ** 0.5
        d_s = (cone_s[0] ** 2 + cone_s[1] ** 2 + cone_s[2] ** 2) ** 0.5
        alpha_cone_c = math.atan2(self.value, d_c)
        alpha_cone_s = math.acos(np.dot(cone_c, cone_s) / (d_c * d_s))
        #        except:
        #            pass
        return alpha_cone_c - alpha_cone_s
