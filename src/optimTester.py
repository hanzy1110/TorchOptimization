#!/usr/bin/env python
from __future__ import division
import os
import numpy as np
from optparse import OptionParser
import matplotlib.pyplot as plt
import json
CORES = 4
os.environ['XLA_FLAGS'] = f'--xla_force_host_platform_device_count={CORES}'

import jax
from jax import jit, pmap
import jax.numpy as jnp
import jaxopt


def processResults(res):
    if isinstance(res, dict):
        return res['u']
    else:
        return res

def dealWithSolver(params, solver, objFunc, maxiter):

    for key in {'options', 'residual_fun', 'method', 'block_prox'}:
        val = params.get(key, 'why')
        if val!='why':
            params.pop(key)

    if solver is jaxopt.ScipyMinimize:
        params.pop('maxiter')
        params['method'] = 'Nelder-Mead'
        # params['options'] = {'hess': jax.hessian(objFunc)}
        # params['hess'] = jax.hessian(objFunc)

    elif solver is jaxopt.GaussNewton:
        params.pop('fun')
        params['residual_fun'] = objFunc

    elif solver is jaxopt.LevenbergMarquardt:
        params.pop('fun')
        params['residual_fun'] = objFunc
        params['maxiter'] = maxiter
    # elif solver is jaxopt.BlockCoordinateDescent:
    #     params['block_prox'] = jaxopt.objective.binary_logreg

    else:
        # try:
        #     params.pop('options')
        #     params.pop('residual_fun')
        # except KeyError as e:
        #     print(e)
        params['fun'] = objFunc

    return params

class jaxEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, jnp.DeviceArray):
            return obj.tolist()
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self,obj)

def pframe(X, p, dim):
    N = X.size // dim
    norms = jnp.sqrt(jnp.sum(X.reshape((-1, dim))**2, 1))
    allsums = jnp.matmul(X.reshape((-1, dim)), X.reshape((-1, dim)).T)
    energy = jnp.abs(allsums / (norms[:, None] * norms[None, :]))**p
    en = jnp.sum(energy) / N**2
    # for l in range(dim):
    #     M = p * energy * (X.reshape((-1, dim))[:, l][None, :]/allsums -
    #                 X.reshape((-1, dim))[:, l][:, None] / norms[:, None]**2)
    #     grad.reshape((-1, dim))[:, l] = np.sum(M, 1) * 2 / N**2.
    return en

class optimTester:
    def __init__(self, optimFunc):
        self.optimFunc = optimFunc
        self.optimDict = {'LBFGS':jaxopt.LBFGS,
                          'BackTrack': jaxopt.BacktrackingLineSearch,
                          'GaussNewton': jaxopt.GaussNewton,
                          'NonLinearCG': jaxopt.NonlinearCG,
                          'Scipy': jaxopt.ScipyMinimize,
                          'LevenbergMarquard': jaxopt.LevenbergMarquardt,
                          'GradDescent': jaxopt.GradientDescent,
                          # 'blockCoordinate': jaxopt.BlockCoordinateDescent
                          }

        solver_params = {'p':3, 'dim':3}
        p, dim, N= 3.0,3, 6

        self.y_true = 0.241202265916660

        self.p = p
        self.N = N
        self.dim = dim

        self.compute_loss = jit(lambda params: self.optimFunc(params['u'],
                                                              p, dim))
        self.loss_grad = jax.grad(self.compute_loss)

    def initPoint(self):
        key = jax.random.PRNGKey(0)
        u = jax.random.normal(key, (self.N, self.dim))
        return (u / np.sqrt(np.sum(u**2, 1))[:, None]).ravel()

    def trainStep(self, params):

        if isinstance(self.solver, jaxopt.BacktrackingLineSearch):
            init_stepsize = 0.1
            init_params = self.solver.init_state(init_stepsize, params)
            sol, state = self.solver.run(init_stepsize, params)

        elif isinstance(self.solver, jaxopt.ScipyMinimize):
            sol,state = self.solver.run(params)
        else:
            init_params = self.solver.init_state(params)
            sol, state = self.solver.run(params)
        try:
            minf = self.compute_loss(sol)
            final_grad = self.loss_grad(sol)
        except Exception as e:
            print(e)
            return None, None, sol, state
        return minf, final_grad, sol, state

    def testOptim(self, solverParams, _iter, maxiter):
        for key, solver in self.optimDict.items():
            print(f'solving for solver: {key}')
            u = self.initPoint()

            if solver is jaxopt.LevenbergMarquardt:
                compute_loss = jit(lambda u: self.optimFunc(u, self.p, self.dim))
                loss_grad = jax.grad(compute_loss)
                solverParams = dealWithSolver(solverParams, solver, compute_loss, maxiter)
                self.solver = solver(**solverParams)
                minf, final_grad, sol, state = self.trainStep(u)
                minf = compute_loss(sol)
                # final_grad = loss_grad(u)
                self.postProcessResults(key,u, minf,
                                        state.gradient, sol,
                                        state, solverParams, _iter)

            else:
                params = {'u':u}
                solverParams = dealWithSolver(solverParams, solver, self.compute_loss, maxiter)
                self.solver = solver(**solverParams)
                minf, final_grad, sol, state = self.trainStep(params)

            self.postProcessResults(key,u, minf,
                                    final_grad, sol,
                                    state, solverParams, _iter)

    def postProcessResults(self, key, u, minf, final_grad, sol, state, config_params, _iter):

        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        dname = os.path.dirname(dname)
        dname = os.path.join(dname,'results')

        try:
            residual = np.log(np.abs(self.y_true-minf))
        except Exception as e:
            print(e)
            residual = np.nan

        sol = processResults(sol)
        final_grad = processResults(final_grad)

        finalDict = {'minValue': minf,
                     'final_grad': final_grad,
                     'solution': sol,
                     'state':state}

        finalDict['residual'] = residual
        finalDict['relDistance'] = jnp.linalg.norm(u-sol)/jnp.linalg.norm(sol)
        finalDict['final_grad_norm'] = jnp.linalg.norm(final_grad)

        if key!='Scipy':
            finalDict['maxiter'] = config_params['maxiter']

        print("minimum value = ", minf)
        print('solver output-->',  sol)
        print('initial point-->', u)
        print(f'Relative Distance between solutions:', finalDict['relDistance'])
        print('log residual-->', residual)
        print('-*-'*10)

        with open(os.path.join(dname, f'{key}_{_iter}.json'), 'w') as file:
            json.dump(finalDict, file, cls=jaxEncoder)
