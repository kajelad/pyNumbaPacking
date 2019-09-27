# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 23:52:31 2019

@author: Jack
"""

import numpy as np
from scipy import sparse
from matplotlib import pyplot as plt


def voigt_map(dim):
    vmap = -np.ones(dim * (dim + 1) // 2, dtype=np.int32)
    vi = 0
    for delta in range(dim):
        for a in range(delta, dim):
            b = a - delta
            cd = 0
            for c in range(dim):
                for d in range(c + 1):
                    if a == c and b == d:
                        vmap[vi] = cd
                    cd += 1
            vi += 1
    return vmap


def voigt_pair_indices(dim):
    indices = np.empty((dim * (dim - 1) // 2, 2), dtype=np.int32)
    voigt = 0
    for delta in range(1, dim):
        for i in range(dim - delta):
            indices[voigt, 0] = i
            indices[voigt, 1] = i + delta
            voigt += 1
    return indices


def get_hessian(packing, stable=False):
    hes = packing.get_real_hessian()
    hes = hes.reshape(
        hes.size // packing.num_dim ** 2,
        packing.num_dim, packing.num_dim
    )
    hes = sparse.bsr_matrix(
        (hes, packing.adj_indices, packing.adj_indptr),
        shape=(
            packing.num_particles * packing.num_dim,
            packing.num_particles * packing.num_dim
        )
    )
    hes += hes.T
    if stable:
        stable_mask = np.repeat(packing.get_stable(), packing.num_dim)
        hes = (
            (hes.tocsr()[stable_mask, :][:, stable_mask])
            .tobsr(blocksize=(packing.num_dim, packing.num_dim))
        )
    hes -= sparse.block_diag(
        [
            hes.data[hes.indptr[i]:hes.indptr[i+1]].sum(axis=0)
            for i in range(hes.indptr.size - 1)
        ],
        format="bsr"
    )
    return hes


def get_affine_tensor(packing):
    asm_data = packing.get_affine_stiffness_matrix()
    astensor = np.empty((packing.num_dim,) * 4, dtype=np.float64)
    abcd = 0
    for a in range(packing.num_dim):
        for b in range(a + 1):
            for c in range(a + 1):
                for d in range(b + 1 if a == c else c + 1):
                    astensor[a, b, c, d] = asm_data[abcd]
                    astensor[b, a, c, d] = asm_data[abcd]
                    astensor[a, b, d, c] = asm_data[abcd]
                    astensor[b, a, d, c] = asm_data[abcd]
                    astensor[c, d, a, b] = asm_data[abcd]
                    astensor[d, c, a, b] = asm_data[abcd]
                    astensor[c, d, b, a] = asm_data[abcd]
                    astensor[d, c, b, a] = asm_data[abcd]
                    abcd += 1
    return astensor


def get_affine_matrix(packing):
    asm_data = packing.get_affine_stiffness_matrix()
    voigt = packing.num_dim * (packing.num_dim + 1) // 2
    asmatrix = np.empty((voigt, voigt), dtype=np.float64)
    abcd = 0
    ab = 0
    for a in range(packing.num_dim):
        for b in range(a + 1):
            cd = 0
            for c in range(a + 1):
                for d in range(b + 1 if a == c else c + 1):
                    asmatrix[ab, cd] = asm_data[abcd]
                    asmatrix[cd, ab] = asm_data[abcd]
                    cd += 1
                    abcd += 1
            ab += 1
    return asmatrix


def get_stiffness_matrix(packing, modes=-1):
    if modes == -1:
        modes = packing.num_particles * packing.num_dim
    stable = np.repeat(packing.get_stable(), packing.num_dim)
    hes = get_hessian(packing, stable=True).todense()
    fs = packing.get_force_stress_matrix().reshape(
        packing.num_particles * packing.num_dim,
        packing.num_dim * (packing.num_dim + 1) // 2
    )[stable, :]
    af = get_affine_matrix(packing)
    E, V = np.linalg.eigh(hes)
    E_inv = 1 / E
    E_inv[:packing.num_dim] = 0
    E_inv[packing.num_dim+modes:] = 0
    stiffness = af - fs.T @ V @ np.diag(E_inv) @ V.T @ fs
    vm = voigt_map(packing.num_dim)
    return stiffness[vm, :][:, vm]


def get_orthotropic_moduli(packing, modes=-1):
    v_ind = voigt_pair_indices(packing.num_dim)
    stiffness = get_stiffness_matrix(packing)
    compliance = np.linalg.inv(stiffness)
    youngs = 1 / compliance.diagonal()[0, :packing.num_dim]
    shear = 1 / compliance.diagonal()[0, packing.num_dim:]
    poisson = -compliance[v_ind[:, 0], v_ind[:, 1]] / youngs[0, v_ind[:, 0]]
    return youngs, shear, poisson


def draw_particles(packing, ax, buffer=0.1, kernel=None):
    basis = packing.basis.reshape(2, 2)
    ax.set_aspect("equal", adjustable="box")
    plt.plot(
        [0, basis[0, 0], basis[0, 0] + basis[1, 0], basis[1, 0], 0],
        [0, basis[0, 1], basis[0, 1] + basis[1, 1], basis[1, 1], 0],
        color=[0, 0, 0]
    )
    pos = packing.positions.reshape(packing.num_particles, packing.num_dim)
    cross = (pos < buffer).astype(np.int32) - (pos > 1-buffer).astype(np.int32)
    stress = np.empty(packing.num_dim**2)
    max_stress = 0
    if kernel is not None:
        for i in range(packing.num_particles):
            packing.single_stress(i, stress)
            max_stress = np.maximum(max_stress, np.abs(np.dot(kernel, stress)))
    for i in range(packing.num_particles):
        packing.single_stress(i, stress)
        if kernel is None:
            facecolor = [0, 0, 1]
        else:
            num = np.dot(stress, kernel)
            facecolor = (
                [1-num/max_stress] * 2 + [1]
            ) if num >= 0 else (
                [1] + [1+num/max_stress] * 2
            )
        ax.add_artist(plt.Circle(
            pos[i] @ basis,
            packing.radii[i],
            facecolor=facecolor, edgecolor=[0, 0, 0]
        ))
        if cross[i, 0]:
            ax.add_artist(plt.Circle(
                (pos[i] + np.array([cross[i, 0], 0])) @ basis,
                packing.radii[i],
                facecolor=facecolor+[0.5], edgecolor=[0, 0, 0, 0.5]
            ))
        if cross[i, 1]:
            ax.add_artist(plt.Circle(
                (pos[i] + np.array([0, cross[i, 1]])) @ basis,
                packing.radii[i],
                facecolor=facecolor+[0.5], edgecolor=[0, 0, 0, 0.5]
            ))
        if np.prod(cross[i]):
            ax.add_artist(plt.Circle(
                (pos[i] + cross[i]) @ basis,
                packing.radii[i],
                facecolor=facecolor+[0.5], edgecolor=[0, 0, 0, 0.5]
            ))


def draw_neighbors(packing, ax, buffer=0.1):
    max_overlap = 0
    for i in range(packing.num_particles):
        for j in packing.neighbors_indices[
                 packing.neighbors_indptr[i]: packing.neighbors_indptr[i+1]
        ]:
            overlap = (
                    packing.radii[i] + packing.radii[j] -
                    packing.distance(i, j)
            )
            max_overlap = np.maximum(max_overlap, overlap)
    basis = packing.basis.reshape(2, 2)
    ax.set_aspect("equal", adjustable="box")
    plt.plot(
        [0, basis[0, 0], basis[0, 0] + basis[1, 0], basis[1, 0], 0],
        [0, basis[0, 1], basis[0, 1] + basis[1, 1], basis[1, 1], 0],
        color=[0, 0, 0]
    )
    pos = packing.positions.reshape(packing.num_particles, packing.num_dim)
    cross = (pos < buffer).astype(np.int32) - (pos > 1-buffer).astype(np.int32)
    for i in range(packing.num_particles):
        for j in packing.neighbors_indices[
                 packing.neighbors_indptr[i]: packing.neighbors_indptr[i+1]
        ]:
            if i > j:
                posi, posj = pos[[i, j], :]
                overlap = (
                        packing.radii[i] + packing.radii[j] -
                        packing.distance(i, j)
                )
                if np.max(np.abs(posj - posi)) < 0.5:
                    plt.plot(
                        *(pos[[i, j], :] @ basis).T,
                        color=[overlap/max_overlap, 1-overlap/max_overlap, 0]
                    )
                if cross[i, 0]:
                    effposi = posi + np.array([cross[i, 0], 0])
                    if np.max(np.abs(posj - effposi)) < 0.5:
                        plt.plot(
                            *(np.array([effposi, posj]) @ basis).T,
                            color=[
                                overlap/max_overlap, 1-overlap/max_overlap, 0
                            ]
                        )
                if cross[i, 1]:
                    effposi = posi + np.array([0, cross[i, 1]])
                    if np.max(np.abs(posj - effposi)) < 0.5:
                        plt.plot(
                            *(np.array([effposi, posj]) @ basis).T,
                            color=[
                                overlap/max_overlap, 1-overlap/max_overlap, 0
                            ]
                        )
                if np.prod(cross[i]):
                    effposi = posi + cross[i]
                    if np.max(np.abs(posj - effposi)) < 0.5:
                        plt.plot(
                            *(np.array([effposi, posj]) @ basis).T,
                            color=[
                                overlap/max_overlap, 1-overlap/max_overlap, 0
                            ]
                        )
                if cross[j, 0]:
                    effposj = posj + np.array([cross[j, 0], 0])
                    if np.max(np.abs(effposj - posi)) < 0.5:
                        plt.plot(
                            *(np.array([posi, effposj]) @ basis).T,
                            color=[
                                overlap/max_overlap, 1-overlap/max_overlap, 0
                            ]
                        )
                if cross[j, 1]:
                    effposj = posj + np.array([0, cross[j, 1]])
                    if np.max(np.abs(effposj - posi)) < 0.5:
                        plt.plot(
                            *(np.array([posi, effposj]) @ basis).T,
                            color=[
                                overlap/max_overlap, 1-overlap/max_overlap, 0
                            ]
                        )
                if np.prod(cross[j]):
                    effposj = posj + cross[j]
                    if np.max(np.abs(effposj - posi)) < 0.5:
                        plt.plot(
                            *(np.array([posi, effposj]) @ basis).T,
                            color=[
                                overlap/max_overlap, 1-overlap/max_overlap, 0
                            ]
                        )
