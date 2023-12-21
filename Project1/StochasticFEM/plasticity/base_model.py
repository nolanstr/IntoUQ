import jax
import jax.numpy as np
import os
import matplotlib.pyplot as plt
import time
import pickle
from scipy.stats import norm

from jax_fem.problem import Problem
from jax_fem.solver import solver
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import box_mesh, get_meshio_cell_type, Mesh



class Plasticity(Problem):

    def custom_init(self):
        self.fe = self.fes[0]
        self.epsilons_old = np.zeros((len(self.fe.cells), self.fe.num_quads, self.fe.vec, self.dim))
        self.plastic_strain = np.zeros((len(self.fe.cells), self.fe.num_quads, self.fe.vec, self.dim))
        self.sigmas_old = np.zeros_like(self.epsilons_old)

    def set_material_properties(self, E, nu, sig0, b, Q_inf):
        self.E = E
        self.nu = nu
        self.sig0 = sig0
        self.b = b
        self.Q_inf = Q_inf
        self.internal_vars = [self.sigmas_old, self.epsilons_old, self.sig0, self.plastic_strain]

    def get_tensor_map(self):
        _, stress_return_map = self.get_maps()
        return stress_return_map

    def get_maps(self, return_all=False):
        def safe_sqrt(x):
            return np.where(x > 0., np.sqrt(x), 0.)

        def safe_divide(x, y):
            return np.where(y == 0., 0., x/y)

        def strain(u_grad):
            epsilon = 0.5 * (u_grad + u_grad.T)
            return epsilon

        def stress(epsilon):
            mu = self.E / (2. * (1. + self.nu))
            lmbda = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
            sigma = lmbda * np.trace(epsilon) * np.eye(self.dim) + 2 * mu * epsilon
            return sigma

        def stress_return_map(u_grad, sigma_old, epsilon_old, sig0, plastic_strain):
            epsilon_crt = strain(u_grad)
            epsilon_inc = epsilon_crt - epsilon_old
            sigma_trial = stress(epsilon_inc) + sigma_old
            s_dev = sigma_trial - 1. / self.dim * np.trace(sigma_trial) * np.eye(self.dim)
            s_norm = safe_sqrt(3. / 2. * np.sum(s_dev * s_dev))

            # Compute volumetric and equivalent plastic strain
            vol = np.sum(self.fe.JxW)
            volumetric_plastic_strain = np.sum(plastic_strain.copy().reshape(
                -1, self.fe.vec, self.dim) * self.fe.JxW.reshape(-1)[:, None, None], 0) / vol
            equivalent_plastic_strain = np.sqrt((2/3) * np.sum(volumetric_plastic_strain ** 2))

            # Isotropic hardening
            sig0 += self.Q_inf * (1 - np.exp(-self.b * equivalent_plastic_strain)) 

            # Compute yield function
            if isinstance(sig0, np.ndarray) and sig0.ndim>0:
                sig0 = sig0[0,0].item()
            f_yield = s_norm - sig0
            f_yield_plus = np.where(f_yield > 0., f_yield, 0.)

            # Update plastic strain
            updated_plastic_strain = plastic_strain + safe_divide(f_yield_plus, (2.0 * self.Q_inf))

            # Update stress
            updated_sigma = sigma_trial - safe_divide(f_yield_plus * s_dev, s_norm)
            try:
                if sig0 == np.nan:
                    import pdb;pdb.set_trace()
            except:
                pass
            if return_all:
                return updated_sigma, updated_plastic_strain, sig0
            return updated_sigma

        return strain, stress_return_map

    def stress_strain_fns(self):
        strain, stress_return_map = self.get_maps(return_all=True)
        vmap_strain = jax.vmap(jax.vmap(strain))
        vmap_stress_return_map = jax.vmap(jax.vmap(stress_return_map))
        return vmap_strain, vmap_stress_return_map

    def update_stress_strain(self, sol):
        u_grads = self.fe.sol_to_grad(sol)
        vmap_strain, vmap_stress_rm = self.stress_strain_fns()

        # Unpack the tuple returned by vmap_stress_rm
        shape = (u_grads.shape[0], u_grads.shape[1])
        updated_sigma, updated_plastic_strain, updated_sig0 = vmap_stress_rm(u_grads, self.sigmas_old, self.epsilons_old, 
                np.full(shape, self.sig0), self.plastic_strain)
        # Update internal variables
        self.sigmas_old = updated_sigma
        self.epsilons_old = vmap_strain(u_grads)
        self.sig0 = updated_sig0[0,0].item()
        self.plastic_strain = updated_plastic_strain
        self.internal_vars = [self.sigmas_old, self.epsilons_old, self.sig0, self.plastic_strain]

    def compute_avg_stress(self):
        """For post-processing only: Compute volume averaged stress."""
        sigma = np.sum(self.sigmas_old.reshape(-1, self.fe.vec, self.dim) * self.fe.JxW.reshape(-1)[:, None, None], 0)
        vol = np.sum(self.fe.JxW)
        avg_sigma = sigma / vol
        return avg_sigma

    def compute_avg_strain(self):
        """For post-processing only: Compute volume averaged strain."""
        strain = np.sum(self.epsilons_old.reshape(-1, self.fe.vec, self.dim) * self.fe.JxW.reshape(-1)[:, None, None], 0)
        vol = np.sum(self.fe.JxW)
        avg_strain = strain / vol
        return avg_strain

    def compute_avg_plastic_strain(self):
        """For post-processing only: Compute volume averaged strain."""
        strain = np.sum(self.plastic_strain.reshape(-1, self.fe.vec, self.dim) * self.fe.JxW.reshape(-1)[:, None, None], 0)
        vol = np.sum(self.fe.JxW)
        avg_strain = strain / vol
        return avg_strain



def run_model(E, sig0, filename, plotname):

    time_start = time.time()

    def top(point):
        return np.isclose(point[2], Lz, atol=1e-5)

    def bottom(point):
        return np.isclose(point[2], 0., atol=1e-5)

    def dirichlet_val_bottom(point):
        return 0.

    def get_dirichlet_top(disp):
        def val_fn(point):
            return disp
        return val_fn

    ele_type = 'HEX8'
    cell_type = get_meshio_cell_type(ele_type)
    data_dir = os.path.join(os.path.dirname(__file__), 'data')

    Lx, Ly, Lz = 10., 10., 10.
    meshio_mesh = box_mesh(Nx=10, Ny=10, Nz=10, Lx=Lx, Ly=Ly, Lz=Lz, data_dir=data_dir, ele_type=ele_type)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

    disps = np.hstack((np.linspace(0., 0.02, 15)))#, np.linspace(0.13, 0., 8)))

    location_fns = [bottom, top]
    value_fns = [dirichlet_val_bottom, get_dirichlet_top(disps[0])]
    vecs = [2, 2]

    dirichlet_bc_info = [location_fns, vecs, value_fns]

    problem = Plasticity(mesh, vec=3, dim=3, dirichlet_bc_info=dirichlet_bc_info) 
    problem.set_material_properties(E=E, nu=0.3, sig0=sig0, b=4.4, Q_inf=25.6e3)
    avg_stresses = []
    avg_strains = []
    avg_plastic_strains = []

    for i, disp in enumerate(disps):
        print(f"\nStep {i} in {len(disps)}, disp = {disp}")
        dirichlet_bc_info[-1][-1] = get_dirichlet_top(disp)
        problem.fe.update_Dirichlet_boundary_conditions(dirichlet_bc_info)
        sol_list = solver(problem, use_petsc=True)
        problem.update_stress_strain(sol_list[0])
        print(problem.sig0)
        avg_stress = problem.compute_avg_stress()
        avg_strain = problem.compute_avg_strain()
        avg_plastic_strain = problem.compute_avg_plastic_strain()
        avg_stresses.append(avg_stress)
        avg_strains.append(avg_strain)
        avg_plastic_strains.append(avg_plastic_strain)

    time_end = time.time() - time_start
    avg_stresses = np.array(avg_stresses)
    avg_strains = np.array(avg_strains)
    avg_plastic_strains = np.array(avg_plastic_strains)

    data = {"material properties":{"E":E, "nu":0.3, "sig0":sig0},
            "displacements":disps,
            "average stresses":avg_stresses,
            "average strains":avg_strains,
            "average plastic strains":avg_plastic_strains,
            "computational time": time_end}
    FILE = open(f"model_outputs/{filename}.pkl", "wb")
    pickle.dump(data, FILE)
    FILE.close()

    fig = plt.figure(figsize=(10, 8))
    plt.plot(avg_strains[:,2,2], avg_stresses[:, 2, 2], color='red', marker='o', markersize=8, linestyle='-') 
    plt.xlabel(r'Displacement of top surface [mm]', fontsize=20)
    plt.ylabel(r'Volume averaged stress (z-z) [MPa]', fontsize=20)
    plt.tick_params(labelsize=18)
    plt.savefig(f"./plots/{plotname}")


if __name__ == "__main__":
    coeff_var = 0.01
    E_mu = 207e3
    E_dist = norm(loc=E_mu, scale=E_mu*coeff_var)
    sig0_mu = 285.
    sig0_dist = norm(loc=sig0_mu, scale=sig0_mu*coeff_var)
    
    N = 100
    for i in range(N):
        E = E_dist.rvs()
        sig0 = sig0_dist.rvs()
        tag = str(i+1).zfill(3)
        filename = f"output_{tag}"
        plotname = f"output_{tag}"
        run_model(E, sig0, filename, plotname)
