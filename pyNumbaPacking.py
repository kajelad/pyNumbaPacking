# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 13:05:29 2018

@author: Jack
"""
import numpy as np
import numba


@numba.jit(numba.float64(numba.int32), nopython=True)
def half_integer_gamma(x):
    """
    halfIntegerGamma(x) = gamma(x / 2)
    x must be an integer
    """
    result = np.sqrt(np.pi) if x % 2 else 1
    for i in range(2 - x % 2, x, 2):
        result *= i / 2
    return result


@numba.jit(numba.float64(numba.int32), nopython=True)
def sphere_volume_constant(d):
    return np.pi ** (d / 2) / half_integer_gamma(d + 2)


@numba.jit(
    numba.float64(
        numba.float64[:], numba.boolean[:], numba.boolean[:], numba.int32
    ),
    nopython=True
)
def cofactor(A, mask1, mask2, n):
    """
    determines the partial determinant based on a series of index masks

    args:
        A (float64[:], n**2): square matrix to evaluate, flattened
        mask1, mask2 (bool[:], n): masks for included rows/columns
        n (int32): size of matrix.
    """
    # Finds the first unmasked index
    for i in range(n):
        if mask1[i]:
            break
    else:
        return 1
    mask1[i] = False
    total = 0
    sign = 1
    for j in range(n):
        if mask2[j]:
            mask2[j] = False
            total += sign * A[i * n + j] * cofactor(A, mask1, mask2, n)
            mask2[j] = True
            sign = -sign
    mask1[i] = True
    return total


@numba.jit(
    numba.void(numba.float64[:], numba.float64[:], numba.int32),
    nopython=True
)
def adjugate_transpose(A, A_adjt, n):
    """
    Computes the adjugate transpose (elementwise derivative of determinant)
    of a matrix

    args:
        A (float64[:], n**2): square matrix, flattened
        A_inv (float64[:], n**2): output matrix, flattened
        n (int32): size of matrices.
    """
    mask1 = np.ones(n, dtype=np.bool8)
    mask2 = np.ones(n, dtype=np.bool8)
    for i in range(n):
        for j in range(n):
            mask1[i] = False
            mask2[j] = False
            A_adjt[i*n+j] = (
                (1 - 2 * ((i + j) % 2)) *
                cofactor(A, mask1, mask2, n)
            )
            mask1[i] = True
            mask2[j] = True


@numba.jit(numba.float64(numba.float64[:], numba.int32), nopython=True)
def determinant(A, n):
    """
    computes the determinant of a flattened n x n matrix A

    args:
        A (float64[:], n**2): square matrix to evaluate, flattened
        n (int32): size of matrix.
    """
    mask1 = np.ones(n, dtype=np.bool8)
    mask2 = np.ones(n, dtype=np.bool8)
    return cofactor(A, mask1, mask2, n)


@numba.jit(
    numba.void(numba.float64[:], numba.float64[:], numba.int32),
    nopython=True
)
def inverse(A, A_inv, n):
    """
    computes the inverse of a flattened n x n matrix A

    args:
        A (float64[:], n**2): square matrix to invert, flattened
        A_inv (float64[:], n**2): output matrix, flattened
        n (int32): size of matrices.
    """
    mask1 = np.ones(n, dtype=np.bool8)
    mask2 = np.ones(n, dtype=np.bool8)
    sign = 1
    det = determinant(A, n)
    for i in range(n):
        for j in range(n):
            mask1[i] = False
            mask2[j] = False
            A_inv[j * n + i] = sign * cofactor(A, mask1, mask2, n) / det
            mask1[i] = True
            mask2[j] = True
            sign = -sign


@numba.jitclass([
    ("num_dim", numba.int32),
    ("num_particles", numba.int32),
    ("volume_constant", numba.float64),
    ("stiffness", numba.float64),

    ("basis", numba.float64[:]),
    ("basis_mass", numba.float64),
    ("basis_velocities", numba.float64[:]),
    ("basis_forces", numba.float64[:]),
    ("metric", numba.float64[:]),
    # ("cometric", numba.float64[:]),

    # ("moments", numba.float64[:]),

    ("masses", numba.float64[:]),
    ("positions", numba.float64[:]),
    ("velocities", numba.float64[:]),
    ("forces", numba.float64[:]),

    ("rad_masses", numba.float64[:]),
    ("radii", numba.float64[:]),
    ("rad_velocities", numba.float64[:]),
    ("rad_forces", numba.float64[:]),

    # ("orientation", numba.float64[:]),
    # ("angular_velocities", numba.float64[:]),
    # ("torques", numba.float64[:]),

    ("neighbors_indptr", numba.int32[:]),
    ("neighbors_indices", numba.int32[:]),

    ("adj_indptr", numba.int32[:]),
    ("adj_indices", numba.int32[:]),
])
class Packing():
    """
    Packing object

    system attributes:
        num_dim (int32): number of dimensions. Fixed value
        num_particles (int32): number of dimensions. Fixed value
        volume_constant (float64): precomputed sphere colume constant. Fixed
            value.
        basis (float64[:], num_dim x num_dim): basis vectors of unit cell.
        basis_mass (float64): scalar "inertia" of the elements of the
            basis vectors. by default, this is half of the system mass.
        basis_velocities (float64[:], num_dim x num_dim): time rate of change
            of elements of basis matrix.
        basis_forces (float64[:], num_dim x num_dim): effective "force" on each
            element of the basis matrix
        metric (float64[:], num_dim x num_dim): Controvariant metric tensor.
            Fixed shape
        cometric (float64[:], num_dim x num_dim): Covariant metric tensor.
            Fixed shape
        stiffness (float64): Contact stiffness.
    particle attributes:
        masses (float64[:], num_particles): particle masses
        positions (float64[:], num_particles x num_dim): Particle positions,
            parameterized by unit cube, Fixed shape
        velocities (float64[:], num_particles x num_dim): Time derivative of
            position array, Fixed shape
        forces (float64[:], num_particles x num_dim): Net forces, Fixed shape
        rad_masses (float64[:], num_particles): generalized masses of particle
            radii
        radii (float64[:], num_particles): particle radii
        rad_velocities (float64[:], num_particle) generalized velocities
            of paticle radii
        rad_forces (float64[:], num_particle) generalized forces
            on paticle radii
    auxillary attributes:
        neighbors (CSR pattern): particle neighbor adjacency that covers
            contacts
        adj (CSR pattern): particle subset of neighbors consisting of actually
            interacting particles with i<j

    Notes:

    CSR attributes are stored as 3 values, and are runtime adjustable in
    number of nonzero values. These values are:
        NAME_indptr (int32[:], num_rows + 1): Start end end of each row.
        NAME_indices (int32[:], nnz): column indices for each value
        NAME (TYPE[:], nnz): Nonzero values
    Pattern CSR encode boolean matrices, all nonzero values are assumed to be
    True.

    ARPACK attributes are stored as 4 values, and are all runtime adjustable in
    row size. These arrays are:
        NAME_maxrow (int32): max row size, adjusted as needed
        NAME_rows (int32[:], NUM_ROWS): number of nonzero elements per row.
        NAME_indices (int32[:], NUM_ROWS x NAME_maxrow), indices of data values
        NAME (TYPE[:], NUM_ROWS x NAME_maxrow): Values, padded
    Pattern ARPACK  encode boolean matrices, all nonzero values are assumed to
    be True.
    """
    def __init__(self, num_dim, num_particles):
        self.num_dim = num_dim
        self.num_particles = num_particles
        self.volume_constant = sphere_volume_constant(self.num_dim)
        self.stiffness = 1.0

        self.masses = np.ones(
            self.num_particles*self.num_dim, dtype=np.float64
            )
        self.positions = np.random.rand(self.num_particles * self.num_dim)
        self.velocities = np.zeros(
            self.num_particles*self.num_dim, dtype=np.float64
        )
        self.forces = np.zeros(
            self.num_particles*self.num_dim, dtype=np.float64
        )

        self.basis_mass = np.sum(self.masses) / 2
        self.basis = np.zeros(self.num_dim**2, dtype=np.float64)
        self.basis_velocities = np.zeros(self.num_dim**2, dtype=np.float64)
        self.basis_forces = np.zeros(self.num_dim**2, dtype=np.float64)
        self.metric = np.zeros(self.num_dim**2, dtype=np.float64)
        # self.cometric = np.zeros(self.num_dim**2, dtype=np.float64)
        for a in range(self.num_dim):
            self.basis[a*self.num_dim+a] = 1
            self.metric[a*self.num_dim+a] = 1
            # self.cometric[a*self.num_dim+a] = 1

        self.rad_masses = np.ones(
            self.num_particles, dtype=np.float64
            )
        self.radii = np.zeros(self.num_particles, dtype=np.float64)
        self.rad_velocities = np.zeros(self.num_particles, dtype=np.float64)
        self.rad_forces = np.zeros(self.num_particles, dtype=np.float64)
        """
        self.orientations = np.zeros(
            self.num_particles * self.num_dim * (self.num_dim - 1) // 2,
            dtype=np.float64
        )
        self.angular_velocities = np.zeros(
            self.num_particles * self.num_dim * (self.num_dim - 1) // 2,
            dtype=np.float64
        )
        self.torques = np.zeros(
            self.num_particles * self.num_dim * (self.num_dim - 1) // 2,
            dtype=np.float64
        )
        """
        self.neighbors_indptr = np.zeros(self.num_particles+1, dtype=np.int32)
        self.neighbors_indices = np.zeros(0, dtype=np.int32)
        self.adj_indptr = np.zeros(self.num_particles+1, dtype=np.int32)
        self.adj_indices = np.zeros(0, dtype=np.int32)

    def basis_update(self):
        """
        updates the quantites dependent on the basis.

        This should be called whenever basis is directly reassigned without
        set_basis.
        """
        self.metric *= 0
        for a in range(self.num_dim):
            for b in range(self.num_dim):
                for c in range(self.num_dim):
                    self.metric[a*self.num_dim+b] += (
                        self.basis[a*self.num_dim+c] *
                        self.basis[b*self.num_dim+c]
                    )

    def set_basis(self, basis):
        """
        sets the basis vectors, and computes the corresponding metric

        args:
            basis (float64[:], num_dim**2): basis vectors, flattened.
        """
        self.basis = basis
        self.basis_update()

    def get_volume(self):
        return determinant(self.basis, self.num_dim)

    def get_phi(self):
        total = 0
        for i in range(self.num_particles):
            total += self.volume_constant * self.radii[i] ** self.num_dim
        return total / self.get_volume()

    def set_phi(self, phi):
        self.radii *= (phi/self.get_phi()) ** (1/self.num_dim)

    def dot(self, u, v):
        total = 0
        for a in range(self.num_dim):
            for b in range(self.num_dim):
                total += u[a] * self.metric[a * self.num_dim + b] * v[b]
        return total

    def displacement(self, i, j, a):
        disp = (
            self.positions[j*self.num_dim+a] -
            self.positions[i*self.num_dim+a]
        )
        disp += 0.5
        disp %= 1
        disp -= 0.5
        return disp

    def distance_squared(self, i, j):
        total = 0
        for a in range(self.num_dim):
            for b in range(self.num_dim):
                total += (
                    self.displacement(i, j, a) *
                    self.metric[a*self.num_dim+b] *
                    self.displacement(i, j, b)
                )
        return total

    def distance(self, i, j):
        return np.sqrt(self.distance_squared(i, j))

    def single_stress(self, i, stress):
        """
        computes the stress on a single particle, in place
        """
        stress[:] = 0
        for ij in range(self.neighbors_indptr[i], self.neighbors_indptr[i+1]):
            j = self.neighbors_indices[ij]
            dist = self.distance(i, j)
            gap = dist - self.radii[i] - self.radii[j]
            if gap < 0:
                prefactor = (
                    self.num_particles * self.stiffness * gap /
                    (2 * self.get_volume() * dist)
                )
                disp = np.zeros(self.num_dim, np.float64)
                for a in range(self.num_dim):
                    for b in range(self.num_dim):
                        disp[a] += (
                            self.basis[a*self.num_dim+b] *
                            self.displacement(i, j, b)
                        )
                for a in range(self.num_dim):
                    for b in range(self.num_dim):
                        stress[a*self.num_dim+b] += (
                            prefactor * disp[a] * disp[b]
                        )

    def single_stress_component(self, i, kernel):
        """
        computes the stress on a single particle dotted with a kernel in order
        to extract the pressure, simple shear, etc.
        """
        result = 0
        for ij in range(self.neighbors_indptr[i], self.neighbors_indptr[i+1]):
            j = self.neighbors_indices[ij]
            dist = self.distance(i, j)
            gap = dist - self.radii[i] - self.radii[j]
            if gap < 0:
                prefactor = (
                    self.num_particles * self.stiffness * gap /
                    (2 * self.get_volume() * dist)
                )
                disp = np.zeros(self.num_dim, np.float64)
                for a in range(self.num_dim):
                    for b in range(self.num_dim):
                        disp[a] += (
                            self.basis[a*self.num_dim+b] *
                            self.displacement(i, j, b)
                        )
                for a in range(self.num_dim):
                    for b in range(self.num_dim):
                        result += (
                            kernel[a*self.num_dim+b] *
                            prefactor * disp[a] * disp[b]
                        )
        return result

    def single_pressure(self, i):
        """
        computes the pressure on a single particle
        """
        pressure = 0
        for ij in range(self.neighbors_indptr[i], self.neighbors_indptr[i+1]):
            j = self.neighbors_indices[ij]
            dist = self.distance(i, j)
            gap = dist - self.radii[i] - self.radii[j]
            if gap < 0:
                prefactor = (
                    -self.num_particles * self.stiffness * gap /
                    (2 * self.num_dim * self.get_volume() * dist)
                )
                for a in range(self.num_dim):
                    disp = 0
                    for b in range(self.num_dim):
                        disp += (
                            self.basis[a*self.num_dim+b] *
                            self.displacement(i, j, b)
                        )
                    pressure += prefactor * disp ** 2
        return pressure

    def single_pure_shear(self, i):
        """
        computes the pressure on a single particle
        """
        pressure = 0
        for ij in range(self.neighbors_indptr[i], self.neighbors_indptr[i+1]):
            j = self.neighbors_indices[ij]
            dist = self.distance(i, j)
            gap = dist - self.radii[i] - self.radii[j]
            if gap < 0:
                prefactor = (
                    -self.num_particles * self.stiffness * gap /
                    (2 * self.get_volume() * dist)
                )
                for a in range(self.num_dim):
                    disp = 0
                    for b in range(self.num_dim):
                        disp += (
                            self.basis[a*self.num_dim+b] *
                            self.displacement(i, j, b)
                        )
                    pressure += prefactor * disp ** 2 / (
                        1 if a == 1 else -self.num_dim + 1
                    )
        return pressure

    def calc_neighbors(self, cut_distance):
        """
        recalculates the neighbor matrix to contain all particle pairs within
        cut_distance of contact
        """
        # Spatial cell binning ------------------------------------------------
        # compute the proper cell size in each dimension
        # TODO: Compute the proper cell size, given shearing possibilities.
        cells_shape = np.empty(self.num_dim, dtype=np.int32)
        crit_distance = np.max(self.radii) * 2 + cut_distance
        for a in range(self.num_dim):
            cells_shape[a] = np.maximum(
                1, np.sqrt(self.metric[a*self.num_dim+a]) // crit_distance
            )
        num_cells = np.prod(cells_shape)
        # computes the cell index of each particle
        particle_cell = np.zeros(self.num_particles, dtype=np.int32)
        # Iterate through the C style flattening of the d-dimensional array
        # of cells
        stride = 1
        for a in range(self.num_dim-1, -1, -1):
            for i in range(self.num_particles):
                particle_cell[i] += stride * int(
                        self.positions[i*self.num_dim+a] * cells_shape[a]
                )
            stride *= cells_shape[a]
        # cell_particle is a temporary CSR pattern array that inverts
        # particle_cell
        cell_particle_indptr = np.zeros(num_cells + 1, dtype=np.int32)
        cell_particle_indices = np.empty(self.num_particles, dtype=np.int32)
        for i in range(self.num_particles):
            cell_particle_indptr[particle_cell[i]+1] += 1
        for i in range(num_cells):
            cell_particle_indptr[i+1] += cell_particle_indptr[i]
        for i in range(self.num_particles):
            cell_particle_indices[cell_particle_indptr[particle_cell[i]]] = i
            cell_particle_indptr[particle_cell[i]] += 1
        for i in range(num_cells + 1, 0, -1):
            cell_particle_indptr[i] = cell_particle_indptr[i-1]
        cell_particle_indptr[0] = 0

        # neighbor array size determination -----------------------------------
        self.neighbors_indptr *= 0
        multi_index = np.empty(self.num_dim, dtype=np.int32)
        kernel = np.empty(self.num_dim, dtype=np.int32)
        for i in range(self.num_particles):
            index = particle_cell[i]
            # unravel the current cell index
            for a in range(self.num_dim-1, -1, -1):
                multi_index[a] = index % cells_shape[a]
                index //= cells_shape[a]
            kernel.fill(-1)
            while True:
                # Get the current cell of interest
                cell = 0
                stride = 1
                for a in range(self.num_dim-1, -1, -1):
                    cell += stride * (
                        (multi_index[a] + kernel[a]) % cells_shape[a]
                    )
                    stride *= cells_shape[a]
                # Iterate through potential neighbors
                for q in range(
                    cell_particle_indptr[cell], cell_particle_indptr[cell+1]
                ):
                    j = cell_particle_indices[q]
                    if (
                        i != j and
                        self.distance(i, j) <
                        self.radii[i] + self.radii[j] + cut_distance
                    ):
                        self.neighbors_indptr[i + 1] += 1
                # Incriment the kernel
                for a in range(self.num_dim-1, -1, -1):
                    if kernel[a] == 1:
                        kernel[a] = -1
                    else:
                        kernel[a] += 1
                        break
                else:
                    break
        for i in range(self.num_particles):
            self.neighbors_indptr[i+1] += self.neighbors_indptr[i]

        # populating neighbor array -------------------------------------------
        self.neighbors_indices = np.zeros(
            self.neighbors_indptr[self.num_particles], dtype=np.int32
        )
        for i in range(self.num_particles):
            index = particle_cell[i]
            # unravel the current cell index
            for a in range(self.num_dim-1, -1, -1):
                multi_index[a] = index % cells_shape[a]
                index //= cells_shape[a]
            kernel.fill(-1)
            while True:
                # Get the current cell of interest
                cell = 0
                stride = 1
                for a in range(self.num_dim-1, -1, -1):
                    cell += stride * (
                        (multi_index[a] + kernel[a]) % cells_shape[a]
                    )
                    stride *= cells_shape[a]
                # Iterate through potential neighbors
                for q in range(
                    cell_particle_indptr[cell], cell_particle_indptr[cell+1]
                ):
                    j = cell_particle_indices[q]
                    if (
                        i != j and
                        self.distance(i, j) <
                        self.radii[i] + self.radii[j] + cut_distance
                    ):
                        self.neighbors_indices[self.neighbors_indptr[i]] = j
                        self.neighbors_indptr[i] += 1
                # Incriment the kernel
                for a in range(self.num_dim-1, -1, -1):
                    if kernel[a] == 1:
                        kernel[a] = -1
                    else:
                        kernel[a] += 1
                        break
                else:
                    break
        for i in range(self.num_particles, 0, -1):
            self.neighbors_indptr[i] = self.neighbors_indptr[i-1]
        self.neighbors_indptr[0] = 0

    def calc_adj(self):
        """
        updates adj_indptr and adj_indices to reflect current overlaps
        """
        self.adj_indptr[0] = 0
        for i in range(self.num_particles):
            self.adj_indptr[i+1] = self.adj_indptr[i]
            for ij in range(
                self.neighbors_indptr[i], self.neighbors_indptr[i+1]
            ):
                j = self.neighbors_indices[ij]
                if i < j:
                    self.adj_indptr[i+1] += 1
        self.adj_indices = np.empty(
            self.adj_indptr[self.num_particles], dtype=np.int32
        )
        self.adj_indptr[0] = 0
        for i in range(self.num_particles):
            self.adj_indptr[i+1] = self.adj_indptr[i]
            for ij in range(
                self.neighbors_indptr[i], self.neighbors_indptr[i+1]
            ):
                j = self.neighbors_indices[ij]
                if i < j:
                    self.adj_indices[self.adj_indptr[i+1]] = j
                    self.adj_indptr[i+1] += 1

    def calc_forces(self):
        self.forces *= 0
        for i in range(self.num_particles):
            for k in range(
                self.neighbors_indptr[i], self.neighbors_indptr[i+1]
            ):
                j = self.neighbors_indices[k]
                dist = self.distance(i, j)
                gap = dist - self.radii[i] - self.radii[j]
                if gap < 0:
                    prefactor = -self.stiffness * gap / dist
                    for a in range(self.num_dim):
                        for b in range(self.num_dim):
                            self.forces[i*self.num_dim+a] += (
                                prefactor * self.basis[b*self.num_dim+a] *
                                self.displacement(j, i, b)
                            )

    def calc_basis_forces(self, pressure):
        """
        computes the forces on the basis vectors of the boundary conditions,
        subject to a global pressure

        args:
            pressure (float64): systemwide pressure. Applies a force based on
                the energy
                U = U_el + Pdet(b)
                where U_el is the elastic potential, and b is the basis (and
                thus det(b) is the volume).
        """
        # The pressure contributes a force of -adj^T(b)
        adjugate_transpose(self.basis, self.basis_forces, self.num_dim)
        self.basis_forces *= -pressure
        # The contribution from the harmonic potential
        for i in range(self.num_particles):
            for k in range(
                self.neighbors_indptr[i], self.neighbors_indptr[i+1]
            ):
                j = self.neighbors_indices[k]
                dist = self.distance(i, j)
                gap = dist - self.radii[i] - self.radii[j]
                if gap < 0:
                    prefactor = -self.stiffness * gap / dist / 2
                    for a in range(self.num_dim):
                        for b in range(self.num_dim):
                            for c in range(self.num_dim):
                                self.basis_forces[a*self.num_dim+c] += (
                                    prefactor *
                                    self.displacement(i, j, a) *
                                    self.displacement(i, j, b) *
                                    self.basis[b*self.num_dim+c]
                                )

    def calc_rad_forces(self, kernel, aging_rate, rad_force):
        self.rad_forces[:] = 0
        for i in range(self.num_particles):
            self.rad_forces[i] += rad_force - (
                aging_rate * self.single_stress_component(i, kernel)
            )

    def calc_hessian_block(self, i, j, block):
        """
        computes a single nondiagonal block of the Hessian
        """
        block[:] = 1
        dist = 0
        for a in range(self.num_dim):
            norm = 0
            for b in range(self.num_dim):
                norm += self.basis[a*self.num_dim+b]*self.displacement(i, j, b)
            dist += norm ** 2
            for b in range(self.num_dim):
                block[a*self.num_dim+b] *= norm
                block[b*self.num_dim+a] *= norm
        block *= -(self.radii[i] + self.radii[j]) / dist
        dist = np.sqrt(dist)
        block[::self.num_dim+1] += (self.radii[i] + self.radii[j] - dist)
        block *= self.stiffness / (2 * dist)

    def get_real_hessian(self):
        """
        Computes the Hessian matrix, extended by d(d+1)/2 affine strain
        parameters
        """
        self.calc_adj()
        hessian = np.empty(
            self.adj_indptr[self.num_particles] * self.num_dim ** 2,
            dtype=np.float64
        )
        for i in range(self.num_particles):
            for ij in range(
                self.neighbors_indptr[i], self.neighbors_indptr[i+1]
            ):
                j = self.neighbors_indices[ij]
                self.calc_hessian_block(
                    i, j, hessian[ij*self.num_dim**2:(ij+1)*self.num_dim**2]
                )
        return hessian

    def get_force_stress_matrix(self):
        """
        Computes the force-stress matrix: the second partial derivatve of
        elastic energy with respect to positions and strain.

        computed in voigt notation: the symmetric components (i>j) are ordered:
        (1,1),
        (2,1),(2,2),
        (3,1),(3,2),(3,3),
        ...
        """
        self.calc_adj()
        # indexing shorthands
        d = self.num_dim
        voigt = d * (d + 1) // 2
        fsmatrix = np.zeros(self.num_particles * d * voigt, dtype=np.float64)
        norm = np.empty(d, dtype=np.float64)
        for i in range(self.num_particles):
            for j in self.adj_indices[
                    self.adj_indptr[i]:self.adj_indptr[i+1]
            ]:
                norm[:] = 0
                for a in range(d):
                    for b in range(d):
                        norm[a] += (
                            self.basis[a*self.num_dim+b] *
                            self.displacement(i, j, b)
                        )
                dist = np.sqrt(np.dot(norm, norm))
                norm /= dist
                for a in range(d):
                    bc = 0
                    for b in range(d):
                        for c in range(b+1):
                            value = self.stiffness * (
                                (self.radii[i] + self.radii[j]) *
                                norm[a] * norm[b] * norm[c] -
                                (self.radii[i] + self.radii[j] - dist) *
                                ((b == c) * norm[a] + (a == c) * norm[b])
                            )
                            fsmatrix[(i*d+a)*voigt+bc] += value
                            fsmatrix[(j*d+a)*voigt+bc] -= value
                            bc += 1
        return fsmatrix

    def get_affine_stiffness_matrix(self):
        """
        computes the affine strain approximation of the stiffness tensor
        in voigt notation, symmetric components are ordered
        (i>j),(k>l),ij>kl
        (1,1;1,1),
        (2,1;1,1),(2,1;2,1),
        (2,2;1,1),(2,2;2,1),(2,2;2,2),
        (3,1;1,1),(3,1;2,1),(3,1;2,2),(3,1;3,1),
        ...
        """
        self.calc_adj()
        voigt = self.num_dim * (self.num_dim + 1) // 2
        asmatrix = np.empty(voigt * (voigt + 1) // 2, dtype=np.float64)
        norm = np.empty(self.num_dim, dtype=np.float64)
        for i in range(self.num_particles):
            for j in self.adj_indices[
                    self.adj_indptr[i]:self.adj_indptr[i+1]
            ]:
                for a in range(self.num_dim):
                    for b in range(self.num_dim):
                        norm += (
                            self.basis[a*self.num_dim+b] *
                            self.displacement(i, j, b)
                        )
                dist = np.sqrt(np.dot(norm, norm))
                norm /= dist
                abcd = 0
                for a in range(self.num_dim):
                    for b in range(a + 1):
                        for c in range(a + 1):
                            for d in range(b + 1 if a == c else c + 1):
                                asmatrix[abcd] = (
                                    self.stiffness * dist *
                                    (self.radii[i] + self.radii[j]) *
                                    norm[a] * norm[b] * norm[c] * norm[d]
                                    )
                                abcd += 1
        return asmatrix

    def FIRE(self, critical_force, max_iterations):
        """
        runs a single FIRE energy minimization

        args:
            critical_force (float64[:]): minimization stops when the maximum
                net force on a particle falls below this threshhold
            max_terations (int32): maximum iterations before minimization will
                stop. Set to -1 for unlimited.
        """
        # FIRE params
        a_start = 0.1
        f_dec = 0.5
        f_inc = 1.1
        f_a = 0.99
        dt = 0.01
        dt_max = 0.1
        N_min = 5
        # neighbor params
        cut_distance = np.max(self.radii) * 2
        min_cut_distance = cut_distance * 0.1
        refine_freq = 64
        last_recalc = 0
        last_positions = np.empty(
            self.num_particles * self.num_dim, dtype=np.float64
        )
        last_positions[:] = self.positions

        self.velocities *= 0
        iteration = 0
        N = 0
        a = a_start
        self.calc_neighbors(cut_distance)
        self.calc_forces()
        while iteration != max_iterations:
            # MD --------------------------------------------------------------
            # single Euler MD step
            self.positions += 0.5 * dt * self.velocities
            self.velocities += self.forces / self.masses
            self.positions += 0.5 * dt * self.velocities
            self.positions %= 1

            self.calc_forces()

            # FIRE ------------------------------------------------------------
            v_dot_f = 0
            f_dot_f = 0
            max_force_squared = 0
            # use the sum of norms as the configuration norm
            for i in range(self.num_particles):
                v_dot_f += self.dot(
                    self.velocities[i*self.num_dim: (i+1)*self.num_dim],
                    self.forces[i*self.num_dim: (i+1)*self.num_dim]
                )
                f = self.dot(
                    self.forces[i*self.num_dim: (i+1)*self.num_dim],
                    self.forces[i*self.num_dim: (i+1)*self.num_dim]
                )
                f_dot_f += f
                max_force_squared = np.maximum(max_force_squared, f)
            # Stop if the max net force is sufficiently small
            if max_force_squared < critical_force ** 2:
                break
            # if the total net force is zero, stop
            if f_dot_f == 0:
                self.velocities *= 0
            # otherwise, bend velocity toward force
            else:
                self.velocities = (
                    (1.0 - a) * self.velocities +
                    a * v_dot_f / f_dot_f * self.forces
                )
            # if velocity points uphill, freeze
            if v_dot_f < 0:
                self.velocities *= 0
                dt *= f_dec
                a = a_start
                N = iteration
            # If enough steps have elapsed since last freeze, speed up
            elif iteration - N > N_min:
                dt = np.minimum(dt * f_inc, dt_max)
                a *= f_a

            # Neighbors -------------------------------------------------------
            # determine distance traveled since last recalculation
            max_displacement = 0
            for i in range(self.num_particles):
                delta = (
                    self.positions[i*self.num_dim: (i+1)*self.num_dim] -
                    last_positions[i*self.num_dim: (i+1)*self.num_dim]
                )
                delta += 0.5
                delta %= 1
                delta -= 0.5
                max_displacement = np.maximum(
                    np.sqrt(self.dot(delta, delta)),
                    max_displacement
                )
            # If enough distace has been trveled to risk new neighbors, update
            if 2 * max_displacement > cut_distance:
                self.calc_neighbors(cut_distance)
                last_recalc = iteration
            # If enough steps have been taken without update, refine the cut
            elif (
                iteration - last_recalc > refine_freq and
                cut_distance > min_cut_distance
            ):
                cut_distance = np.maximum(min_cut_distance, cut_distance*0.5)
                self.calc_neighbors(cut_distance)
                last_recalc = iteration
            iteration += 1
            if not iteration % 10:
                print(iteration, np.sqrt(max_force_squared))
        return iteration

    def const_pressure_FIRE(
        self, critical_force, critical_basis_force, max_iterations, pressure
    ):
        """
        runs a single FIRE energy minimization

        args:
            critical_force (float64): minimization stops when the maximum
                net force on a particle falls below this threshhold
            critical_basis_force (float64): minimization stops when the maximum
                net force on a basis element falls below this threshhold
            max_terations (int32): maximum iterations before minimization will
                stop. Set to -1 for unlimited.
        """
        # FIRE params
        a_start = 0.1
        f_dec = 0.5
        f_inc = 1.1
        f_a = 0.99
        dt = 0.01
        dt_max = 0.1
        N_min = 5
        # neighbor params
        cut_distance = np.max(self.radii) * 2
        min_cut_distance = cut_distance * 0.1
        refine_freq = 64
        last_recalc = 0
        last_positions = np.empty(
            self.num_particles * self.num_dim, dtype=np.float64
        )
        last_positions[:] = self.positions

        self.velocities *= 0
        self.basis_velocities *= 0
        iteration = 0
        N = 0
        a = a_start
        self.calc_neighbors(cut_distance)
        self.calc_forces()
        self.calc_basis_forces(pressure)
        while iteration != max_iterations:
            # MD --------------------------------------------------------------
            # single Euler MD step
            self.positions += 0.5 * dt * self.velocities
            self.basis += 0.5 * dt * self.basis_velocities
            self.basis_update()
            self.velocities += self.forces / self.masses
            self.basis_velocities += self.basis_forces / self.basis_mass
            self.positions += 0.5 * dt * self.velocities
            self.basis += 0.5 * dt * self.basis_velocities
            self.basis_update()
            self.positions %= 1

            self.calc_forces()
            self.calc_basis_forces(pressure)

            # FIRE ------------------------------------------------------------
            v_dot_f = 0
            f_dot_f = 0
            max_force_squared = 0
            # use the sum of norms as the configuration norm
            for i in range(self.num_particles):
                v_dot_f += self.dot(
                    self.velocities[i*self.num_dim: (i+1)*self.num_dim],
                    self.forces[i*self.num_dim: (i+1)*self.num_dim]
                )
                f = self.dot(
                    self.forces[i*self.num_dim: (i+1)*self.num_dim],
                    self.forces[i*self.num_dim: (i+1)*self.num_dim]
                )
                f_dot_f += f
                max_force_squared = np.maximum(max_force_squared, f)
            # repeat the process for the basis
            basis_v_dot_f = 0
            basis_f_dot_f = 0
            for ab in range(self.num_dim**2):
                basis_v_dot_f += (
                    self.basis_velocities[ab] * self.basis_forces[ab]
                )
                basis_f_dot_f += self.basis_forces[ab] ** 2
            # Stop if the max net force is sufficiently small
            if (
                max_force_squared < critical_force ** 2 and
                np.max(np.abs(self.basis_forces)) < critical_basis_force
            ):
                break
            # if the total net force is zero, stop
            if f_dot_f == 0:
                self.velocities *= 0
                # self.basis_velocities *= 0
            # otherwise, bend velocity toward force
            else:
                self.velocities = (
                    (1.0 - a) * self.velocities +
                    a * v_dot_f / f_dot_f * self.forces
                )
                self.basis_velocities = (
                    (1.0 - a) * self.basis_velocities +
                    a * basis_v_dot_f / basis_f_dot_f * self.basis_forces
                )
            # if velocity points uphill, freeze
            if v_dot_f < 0:
                self.velocities *= 0
                dt *= f_dec
                a = a_start
                N = iteration
            # If enough steps have elapsed since last freeze, speed up
            elif iteration - N > N_min:
                dt = np.minimum(dt * f_inc, dt_max)
                a *= f_a

            # Neighbors -------------------------------------------------------
            # determine distance traveled since last recalculation
            max_displacement = 0
            for i in range(self.num_particles):
                delta = (
                    self.positions[i*self.num_dim: (i+1)*self.num_dim] -
                    last_positions[i*self.num_dim: (i+1)*self.num_dim]
                )
                delta += 0.5
                delta %= 1
                delta -= 0.5
                max_displacement = np.maximum(
                    np.sqrt(self.dot(delta, delta)),
                    max_displacement
                )
            # If enough distace has been trveled to risk new neighbors, update
            if 2 * max_displacement > cut_distance:
                self.calc_neighbors(cut_distance)
                last_recalc = iteration
            # If enough steps have been taken without update, refine the cut
            elif (
                iteration - last_recalc > refine_freq and
                cut_distance > min_cut_distance
            ):
                cut_distance = np.maximum(min_cut_distance, cut_distance*0.5)
                self.calc_neighbors(cut_distance)
                last_recalc = iteration

            iteration += 1

        return iteration

    def const_pressure_dyn_rad_FIRE(
        self, critical_force, critical_basis_force, critical_rad_force,
        max_iterations, pressure, kernel, aging_rate, rad_force
    ):
        """
        runs a single FIRE energy minimization

        args:
            critical_force (float64): minimization stops when the maximum
                net force on a particle falls below this threshhold
            critical_basis_force (float64): minimization stops when the maximum
                net force on a basis element falls below this threshhold
            critical_rad_force (float64): minimization stops when the maximum
                net force on a radius falls below this threshhold
            max_terations (int32): maximum iterations before minimization will
                stop. Set to -1 for unlimited.
        """
        # FIRE params
        a_start = 0.1
        f_dec = 0.5
        f_inc = 1.1
        f_a = 0.99
        dt = 0.01
        dt_max = 0.1
        N_min = 5
        # neighbor params
        cut_distance = np.max(self.radii) * 2
        min_cut_distance = cut_distance * 0.1
        refine_freq = 64
        last_recalc = 0
        last_positions = np.empty(
            self.num_particles * self.num_dim, dtype=np.float64
        )
        last_positions[:] = self.positions

        self.velocities *= 0
        self.basis_velocities *= 0
        self.rad_velocities *= 0
        iteration = 0
        N = 0
        a = a_start
        self.calc_neighbors(cut_distance)
        self.calc_forces()
        self.calc_basis_forces(pressure)
        self.calc_rad_forces(kernel, aging_rate, rad_force)
        while iteration != max_iterations:
            # MD --------------------------------------------------------------
            # single Euler MD step
            self.positions += 0.5 * dt * self.velocities
            self.basis += 0.5 * dt * self.basis_velocities
            self.basis_update()
            self.radii += 0.5 * dt * self.rad_velocities
            self.velocities += self.forces / self.masses
            self.basis_velocities += self.basis_forces / self.basis_mass
            self.rad_velocities += self.rad_forces / self.rad_masses
            self.positions += 0.5 * dt * self.velocities
            self.basis += 0.5 * dt * self.basis_velocities
            self.basis_update()
            self.radii += 0.5 * dt * self.rad_velocities
            self.positions %= 1

            self.calc_forces()
            self.calc_basis_forces(pressure)
            self.calc_rad_forces(kernel, aging_rate, rad_force)

            # FIRE ------------------------------------------------------------
            v_dot_f = 0
            f_dot_f = 0
            max_force_squared = 0
            # use the sum of norms as the configuration norm
            for i in range(self.num_particles):
                v_dot_f += self.dot(
                    self.velocities[i*self.num_dim: (i+1)*self.num_dim],
                    self.forces[i*self.num_dim: (i+1)*self.num_dim]
                )
                f = self.dot(
                    self.forces[i*self.num_dim: (i+1)*self.num_dim],
                    self.forces[i*self.num_dim: (i+1)*self.num_dim]
                )
                f_dot_f += f
                max_force_squared = np.maximum(max_force_squared, f)
            # repeat the process for the basis
            basis_v_dot_f = 0
            basis_f_dot_f = 0
            for ab in range(self.num_dim**2):
                basis_v_dot_f += (
                    self.basis_velocities[ab] * self.basis_forces[ab]
                )
                basis_f_dot_f += self.basis_forces[ab] ** 2
            # repeat the process for the radii
            rad_v_dot_f = 0
            rad_f_dot_f = 0
            for i in range(self.num_particles):
                rad_v_dot_f += (
                    self.rad_velocities[i] * self.rad_forces[i]
                )
                rad_f_dot_f += self.rad_forces[i] ** 2
            # Stop if the max net force is sufficiently small
            if (
                max_force_squared < critical_force ** 2 and
                np.max(np.abs(self.basis_forces)) < critical_basis_force and
                np.max(np.abs(self.rad_forces)) < critical_rad_force
            ):
                break
            # if the total net force is zero, stop
            if f_dot_f == 0:
                self.velocities *= 0
                # self.basis_velocities *= 0
            # otherwise, bend velocity toward force
            else:
                self.velocities = (
                    (1.0 - a) * self.velocities +
                    a * v_dot_f / f_dot_f * self.forces
                )
                self.basis_velocities = (
                    (1.0 - a) * self.basis_velocities +
                    a * basis_v_dot_f / basis_f_dot_f * self.basis_forces
                )
                self.rad_velocities = (
                    (1.0 - a) * self.rad_velocities +
                    a * rad_v_dot_f / rad_f_dot_f * self.rad_forces
                )
            # if velocity points uphill, freeze
            if v_dot_f < 0 or basis_v_dot_f < 0 or rad_v_dot_f < 0:
                self.velocities *= 0
                self.basis_velocities *= 0
                self.rad_velocities *= 0
                dt *= f_dec
                a = a_start
                N = iteration
            # If enough steps have elapsed since last freeze, speed up
            elif iteration - N > N_min:
                dt = np.minimum(dt * f_inc, dt_max)
                a *= f_a

            # Neighbors -------------------------------------------------------
            # determine distance traveled since last recalculation
            max_displacement = 0
            for i in range(self.num_particles):
                delta = (
                    self.positions[i*self.num_dim: (i+1)*self.num_dim] -
                    last_positions[i*self.num_dim: (i+1)*self.num_dim]
                )
                delta += 0.5
                delta %= 1
                delta -= 0.5
                max_displacement = np.maximum(
                    np.sqrt(self.dot(delta, delta)),
                    max_displacement
                )
            # If enough distace has been trveled to risk new neighbors, update
            if 2 * max_displacement > cut_distance:
                self.calc_neighbors(cut_distance)
                last_recalc = iteration
            # If enough steps have been taken without update, refine the cut
            elif (
                iteration - last_recalc > refine_freq and
                cut_distance > min_cut_distance
            ):
                cut_distance = np.maximum(min_cut_distance, cut_distance*0.5)
                self.calc_neighbors(cut_distance)
                last_recalc = iteration

            iteration += 1

        return iteration
