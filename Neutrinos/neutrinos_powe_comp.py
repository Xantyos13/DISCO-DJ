# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: discodj-master
#     language: python
#     name: discodj-master
# ---

# %%
# Import modules
from discodj.cosmology.cosmology import Cosmology
import os
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
import jax.numpy as jnp

import jax
from jax import config
jax.config.update("jax_enable_x64", False)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
devices = jax.devices()
device = "gpu" if np.any([d.platform == "gpu" for d in devices]) else "cpu"
print(device)
from discodj import DiscoDJ
print(jax.__version__)

import baccoemu
from classy import Class
from scipy.interpolate import interp1d

# %%
##### COSMOLOGY #####
cosmo = {
    "h": 0.67,
    "H0": 67,
    "ombh2": 0.049*0.67**2,
    "omch2": 0.269984542*0.67**2,
    "As": 2.215e-09,
    "ns": 0.9619,
    "sum_mnu": 0.3,      
    "tau": 0.0952,
    "w0": -1.0,
    "wa": 0.0,
    "z": 0,
    "Tcmb": 2.7255,
}

h = cosmo["h"]
n_s = cosmo["ns"]
A_s  = cosmo["As"]
z = cosmo["z"]
a = 1.0 / (1.0 + z)
sum_mnu = cosmo["sum_mnu"]

omega_nu = cosmo["sum_mnu"]/(93.14*(h**2))
omega_b = cosmo["ombh2"] / h**2
omega_cdm = cosmo["omch2"] / h**2
omega_cdm_nu  = cosmo["omch2"] / h**2  - omega_nu
omega_cold = omega_b + omega_cdm
omega_cold_nu = omega_b + omega_cdm_nu

print("Omega_b=",omega_b)
print("Omega_nu=",omega_nu)
print("Omega_cdm=",omega_cdm)
print("Omega_nu+Omega_cdm_nu=",omega_nu+omega_cdm_nu)
print("Omega_m=",omega_cold+omega_nu)

###### Neutrinos masses ######
mnu_massless = [0,0,0]
mnu_deg = [cosmo["sum_mnu"]/3,cosmo["sum_mnu"]/3,cosmo["sum_mnu"]/3] 

# modes to sample
nmodes = 512
kmin = 1e-4
kmax = 4.9
aexp = a
zout = z
a_array = np.linspace(0.02,a,100)

#k scale
k_h = np.logspace(-3, np.log10(4.9), 600)  # h/Mpc
k_check = np.array([1e-3, 1e-2, 1e-1, 1.0])


# %%
###########################  BACCO MASSLESS   ###########################

bacco_kwargs = dict(
    omega_cold=omega_cold,
    omega_baryon=omega_b,
    A_s=cosmo["As"],
    hubble=h,
    ns=cosmo["ns"],
    neutrino_mass=0,
    w0=cosmo["w0"],
    wa=cosmo["wa"],
    expfactor=a,
    k=k_h,
    cold=True,
)

mp_bacco = baccoemu.Matter_powerspectrum(
    linear=True,
    nonlinear_boost=True,
    baryonic_boost=False,
    verbose=False,
)

sigma8 = mp_bacco.get_sigma8(**bacco_kwargs)
print(sigma8)

k_bacco_lin_h, P_bacco_lin = mp_bacco.get_linear_pk(**bacco_kwargs)
k_bacco_nl_h, P_bacco_nl = mp_bacco.get_nonlinear_pk(**bacco_kwargs, baryonic_boost=False)

pk_ref = np.exp(interp1d(np.log(k_bacco_lin_h), np.log(P_bacco_lin), kind="linear", bounds_error=False, fill_value=np.nan)(np.log(k_h)))
pk_nl_ref = np.exp(interp1d(np.log(k_bacco_nl_h), np.log(P_bacco_nl), kind="linear", bounds_error=False, fill_value=np.nan)(np.log(k_h)))

np.savetxt(f"Pk_cb_massless_z{z}_BACCO.txt",np.column_stack([k_h, pk_ref]),header=f"k[h/Mpc]    Pk[Mpc^3/h^3] (CDM+baryons, z={z}, massless neutrinos)")

###########################  BACCO MASSIVE   ###########################


#Modify the param for the massive neutrinos
bacco_kwargs['omega_cold'] = omega_cold_nu
bacco_kwargs['neutrino_mass'] = cosmo["sum_mnu"]

sigma8_nu = mp_bacco.get_sigma8(**bacco_kwargs)
print(sigma8_nu)

k_bacco_lin_h, P_bacco_lin = mp_bacco.get_linear_pk(**bacco_kwargs)
k_bacco_nl_h, P_bacco_nl = mp_bacco.get_nonlinear_pk(**bacco_kwargs, baryonic_boost=False)

pk_lin = np.exp(interp1d(np.log(k_bacco_lin_h), np.log(P_bacco_lin), kind="linear", bounds_error=False, fill_value=np.nan)(np.log(k_h)))
pk_nl = np.exp(interp1d(np.log(k_bacco_nl_h), np.log(P_bacco_nl), kind="linear", bounds_error=False, fill_value=np.nan)(np.log(k_h)))

fname = f"Pk_cb_sumMnu_{cosmo["sum_mnu"]:.2f}eV_z{z}_BACCO.txt"
np.savetxt(fname,np.column_stack([k_h, pk_lin]),header=f"k[h/Mpc]    Pk[Mpc^3/h^3] (CDM+baryons, z={z}, sum m_nu = {cosmo["sum_mnu"]} eV)")

#%%

###########################   CLASS MASSLESS   ###########################

class_cosmo = Class()

class_params = {
    'output':'dTk vTk mPk',
    "h": h,
    "omega_b": cosmo["ombh2"],
    "omega_cdm": cosmo["omch2"],
    "A_s": cosmo["As"],
    "n_s": cosmo["ns"],
    "tau_reio": cosmo["tau"],
    "N_ncdm": 1,
    'deg_ncdm': 3,
    "m_ncdm": 1e-10/3,
    "N_ur": 0,
    "P_k_max_h/Mpc": 10.0,
    "z_pk": z,
    'T_cmb': 2.7255,
    'YHe': 0.24,
    'T_ncdm': 0.71611,
    'gauge':'newtonian',
    
}

class_cosmo.set(class_params)
class_cosmo.compute()

trans = class_cosmo.get_transfer(z)

k_hmpc = trans['k (h/Mpc)']
k_mpc  = k_hmpc * h         

calH = a * class_cosmo.Hubble(z)  

f_cdm = omega_cdm / (omega_cdm + omega_b)
f_b   = omega_b   / (omega_cdm + omega_b)

d_cb = f_cdm * trans['d_cdm'] + f_b * trans['d_b']
t_tot = trans['t_tot'] 
phi = trans['phi']
zeta = phi - calH * t_tot / k_mpc**2

deltaN_cb = d_cb - 3.0 * zeta + 3.0 * phi

k_pivot = 0.05  

P_R = (2.0 * np.pi**2) / k_mpc**3 * A_s * (k_mpc / k_pivot)**(n_s - 1.0)
ratio = deltaN_cb / d_cb 

P_poisson_cb = np.array([class_cosmo.pk_cb(ki * h, z) * h**3 for ki in k_hmpc])
P_Nm_cb_h = ratio**2 * P_poisson_cb

np.savetxt(f"Pk_cb_massless_z{z}_CLASS.txt",np.column_stack([k_hmpc, P_Nm_cb_h]),header=f"k[h/Mpc]    Pk[Mpc^3/h^3] (CDM+baryons, z={z}, massless neutrinos)")


###########################   CLASS  MASSIVE  ###########################

class_cosmo.struct_cleanup()
class_cosmo.empty()

class_params = {
    'output':'dTk vTk mPk',
    "h": h,
    "omega_b": cosmo["ombh2"],
    "omega_cdm": omega_cdm_nu*h**2,
    "A_s": cosmo["As"],
    "n_s": cosmo["ns"],
    "tau_reio": cosmo["tau"],
    "N_ncdm": 1,
    'deg_ncdm': 3,
    "m_ncdm": cosmo["sum_mnu"]/3,
    "N_ur": 0,
    "P_k_max_h/Mpc": 10.0,
    "z_pk": z,
    'T_cmb': 2.7255,
    'YHe': 0.24,
    'T_ncdm': 0.71611,
    'gauge':'newtonian',
    'perturbations_verbose':1
}

class_cosmo.set(class_params)
class_cosmo.compute()

trans_nu = class_cosmo.get_transfer(z)

k_hmpc = trans_nu['k (h/Mpc)']
k_mpc  = k_hmpc * h         

calH_nu = a * class_cosmo.Hubble(z) 

f_cdm_nu = omega_cdm / (omega_cdm + omega_b)
f_b_nu   = omega_b   / (omega_cdm + omega_b)

d_cb_nu = f_cdm_nu * trans_nu['d_cdm'] + f_b_nu * trans_nu['d_b']
t_tot_nu = trans_nu['t_tot'] 
phi_nu = trans_nu['phi']
zeta_nu = phi_nu - calH_nu * t_tot / k_mpc**2

deltaN_cb_nu = d_cb_nu - 3.0 * zeta_nu + 3.0 * phi_nu

k_pivot = 0.05  

P_R = (2.0 * np.pi**2) / k_mpc**3 * A_s * (k_mpc / k_pivot)**(n_s - 1.0)
ratio_nu = deltaN_cb_nu / d_cb_nu 

P_poisson_cb_nu = np.array([class_cosmo.pk_cb(ki * h, z) * h**3 for ki in k_hmpc]) 
P_Nm_cb_h_nu = ratio_nu**2 * P_poisson_cb_nu


fname = f"Pk_cb_sumMnu_{cosmo["sum_mnu"]:.2f}eV_z{z}_CLASS.txt"
np.savetxt(fname,np.column_stack([k_hmpc, P_Nm_cb_h_nu]),header=f"k[h/Mpc]    Pk[Mpc^3/h^3] (CDM+baryons, z={z}, sum m_nu = {cosmo["sum_mnu"]} eV)")


# %% 

#zlist = np.logspace(np.log10(160), -2, 70)
#tau_list = []
#
#HT3 = []
#
#for z in zlist:
#    class_cosmo.struct_cleanup()
#    class_cosmo.empty()
#
#    class_params = {
#        'output':'dTk vTk mPk',
#        "h": h,
#        "omega_b": cosmo["ombh2"],
#        "omega_cdm": omega_cdm_nu*h**2,
#        "A_s": cosmo["As"],
#        "n_s": cosmo["ns"],
#        "tau_reio": cosmo["tau"],
#        "N_ncdm": 1,
#        'deg_ncdm': 3,
#        "m_ncdm": cosmo["sum_mnu"]/3,
#        "N_ur": 0,
#        "P_k_max_h/Mpc": 10.0,
#        "z_pk": z,
#        'T_cmb': 2.7255,
#        'YHe': 0.24,
#        'T_ncdm': 0.71611,
#        'gauge':'newtonian',
#        'perturbations_verbose':1
#    }
#
#    class_cosmo.set(class_params)
#    class_cosmo.compute()
#    trans = class_cosmo.get_transfer(z)
#    k_hmpc = trans['k (h/Mpc)']
#    k_mpc  = k_hmpc * h          # now in 1/Mpc
#    a    = 1.0 / (1.0 + z)
#    calH = a * class_cosmo.Hubble(z)   # in 1/Mpc
#    f_cdm = omega_cdm / (omega_cdm + omega_b)
#    f_b   = omega_b   / (omega_cdm + omega_b)
#    d_cb = f_cdm * trans['d_cdm'] + f_b * trans['d_b']
#    t_tot = trans['t_tot'] 
#    phi = trans['phi']
#    zeta = phi - calH * t_tot / k_mpc**2
#    deltaN = d_cb - 3.0 * zeta + 3.0 * phi
#
#    D = deltaN[103]/deltaN_f[103]
#
#    HT = d_cb + 3*phi - D*deltaN_f
#
#    HT3.append(HT-3*zeta)
#    bg = class_cosmo.get_background()
#
#HT3 = np.array(HT3)
#tau_of_z = interp1d(
#    bg["z"][::-1],
#    bg["conf. time [Mpc]"][::-1],
#    kind="cubic"
#)
#
#taulist = tau_of_z(zlist)
#
#plt.figure(figsize=(8,6))
#
#k_values = np.array([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
#k_indices = [np.argmin(np.abs(k_hmpc - kv)) for kv in k_values]
#
#for i, (idx, k) in enumerate(zip(k_indices, k_values)):
#    plt.semilogx(taulist, HT3[:, idx], label=fr"$k={k}\,h/\mathrm{{Mpc}}$")
#
#plt.xlabel(r"$\tau\,[\mathrm{Mpc}]$")
#plt.ylabel(r"$H_T-3\zeta$")
#plt.legend(prop={'size': 10},loc=2,ncol=3)
#plt.grid()
#plt.tight_layout()
#plt.show()



# %%
@jax.jit()
def disco_dj_forward_nonlinear(cosmo:Cosmology, input_ps_file:str):
    name = "3D_analysis"
    dim = 3
    precision = "single"
    if precision == "double":
        config.update("jax_enable_x64", True)

    # Define the boxsize and resolution
    boxsize = 200  # in Mpc/h
    res = 128  # the particles live on a Lagrangrian grid of resolution (res)^dim
    dj = DiscoDJ(dim=dim, res=res, name=name, device=device, precision=precision, boxsize=boxsize, cosmo=cosmo)
    print(dj)
    dj = dj.with_timetables()
    pk_state = dj.with_linear_ps(fix_sigma8=False, transfer_function="from_file", filename=input_ps_file)
    white_noise_field = dj.get_ngenic_noise(seed=13)  # a white noise field, defined in real space
    ics = dj.with_ics(pk_state=pk_state, white_noise_space="real", white_noise=white_noise_field)
    delta_ini=jax.numpy.fft.irfftn(-dj.k ** 2 * ics.fphi)

    k_pre_nl, Pk_pre_nl, _ = dj.evaluate_power_spectrum(delta_ini, compute_std=False, bins=80)

    n_order = 2
    stepper = "bullfrog"
    method = "pm"
    res_pm = dj.res
    time_var = "D"
    antialias = 0
    grad_kernel_order = 0
    laplace_kernel_order = 0
    worder = 3
    n_resample = 1
    deconvolve = True
    chunk_size = None  # if you are running out of GPU memory, doing the mass assignment in smaller chunks might help (e.g. chunk_size = dj.res ** dj.dim // 16)


    numsteps = 20  # number of steps to be performed
    a_ini = 0.02  # initial scale factor of the simulation (where it is initialized with LPT)
    a_end = a
    a_list = np.linspace(a_ini, a_end, numsteps, endpoint=True)

    lpt_state = dj.with_lpt(ics, n_order=n_order)
    sim_ini = dj.run_lpt(lpt_state, n_order=n_order, a=a_ini)

    sim_out = dj.run_nbody(sim_ini, a_end=a_end, n_steps=numsteps, res_pm=res_pm,
                                    time_var=time_var, stepper=stepper, method=method, antialias=antialias,
                                    grad_kernel_order=grad_kernel_order, laplace_kernel_order=laplace_kernel_order,
                                    n_resample=n_resample,
                                    deconvolve=deconvolve, chunk_size=chunk_size)

    delta_sim = dj.get_delta_from_pos(sim_out, res= 2*dj.res, n_resample=n_resample,antialias=True)
    k, Pk, _ = dj.evaluate_power_spectrum(delta_sim, bins=dj.res//3, deconvolve=deconvolve, worder=worder)
    return k,Pk, a_end,k_pre_nl, Pk_pre_nl


# %% 
cosmo = Cosmology(Omega_c=omega_cdm,  # cold dark matter content
             Omega_b=omega_b,  # baryonic content (note: Disco-DJ so far only performs N-body simulations, no hydro; this is only used for the linear power spectrum!)
             h=h,  # dimensionless Hubble constant
             n_s=n_s,  # scalar spectral index
             sigma8=sigma8,  # amplitude of matter density fluctuations at a scale of 8 Mpc/h
             mnu = jnp.asarray(mnu_massless)) #Neutrino mass (eV)

cosmo_nu = Cosmology(Omega_c=omega_cdm_nu,  # cold dark matter content
            Omega_b=omega_b,  # baryonic content (note: Disco-DJ so far only performs N-body simulations, no hydro; this is only used for the linear power spectrum!)
            h=h,  # dimensionless Hubble constant
            n_s=n_s,  # scalar spectral index
            sigma8=sigma8_nu,  # amplitude of matter density fluctuations at a scale of 8 Mpc/h
            mnu = jnp.asarray(mnu_deg)) #Neutrino mass (eV)


D = cosmo.compute_unnormed_growth(a_array, 0.02)["Dplus"]
D_nu = cosmo_nu.compute_unnormed_growth(a_array, 0.02)["Dplus"] 


plt.semilogx(a_array,D)
plt.semilogx(a_array,D_nu)


H = cosmo.H(a_array)
H_nu = cosmo_nu.H(a_array)

plt.figure(figsize=(6, 4))
plt.semilogx(a_array, abs(H_nu/H),color='blue',lw=2)  
plt.xlabel(r'Scale factor $a$')
plt.ylabel(r'$H_{\nu}/H$')
plt.grid(True, which='both', ls='--')
plt.tight_layout()
plt.show()

# %%
cosmo = dict(Omega_c=omega_cdm,  # cold dark matter content
             Omega_b=omega_b,  # baryonic content (note: Disco-DJ so far only performs N-body simulations, no hydro; this is only used for the linear power spectrum!)
             h=h,  # dimensionless Hubble constant
             n_s=n_s,  # scalar spectral index
             sigma8=sigma8,  # amplitude of matter density fluctuations at a scale of 8 Mpc/h
             mnu = mnu_massless) #Neutrino mass (eV)



k,Pk, a_end,k_pre_nl, Pk_pre_nl=disco_dj_forward_nonlinear(cosmo,f"Pk_cb_massless_z{z}_CLASS.txt")

# %%
cosmo_nu = dict(Omega_c=omega_cdm_nu,  # cold dark matter content
            Omega_b=omega_b,  # baryonic content (note: Disco-DJ so far only performs N-body simulations, no hydro; this is only used for the linear power spectrum!)
            h=h,  # dimensionless Hubble constant
            n_s=n_s,  # scalar spectral index
            sigma8=sigma8_nu,  # amplitude of matter density fluctuations at a scale of 8 Mpc/h
            mnu = mnu_deg) #Neutrino mass (eV)

D_nu = cosmo_nu.compute_unnormed_growth(a, 0.02)
plot.semilogx(a,D_nu)

k,Pk_nu, a_end,k_pre_nu_nl, Pk_pre_nu_nl=disco_dj_forward_nonlinear(cosmo_nu, f"Pk_cb_sumMnu_{sum_mnu:.2f}eV_z{z}_CLASS.txt")

# %%
Pk = np.exp(interp1d(np.log(k), np.log(Pk), kind="linear", bounds_error=False, fill_value=np.nan)(np.log(k_h)))
Pk_nu =  np.exp(interp1d(np.log(k), np.log(Pk_nu), kind="linear", bounds_error=False, fill_value=np.nan)(np.log(k_h)))

# %%
fig, (ax, axr) = plt.subplots(
    2,
    1,
    figsize=(7, 5),
    sharex=True,
    constrained_layout=True,
    gridspec_kw={"height_ratios": [2, 1]},
)
ax.loglog(k_hmpc, P_Nm_cb_h, color="C1", lw=1.8, label=r"CLASS NM ")
ax.loglog(k_h, pk_ref, color="C2", lw=1.8, ls=":", label=r"BACCO linear ")
ax.loglog(k_h, pk_nl_ref, color="C3", lw=1.8, label=r"BACCO nonlinear")
ax.loglog(k_pre_nl, Pk_pre_nl, color="C4", lw=1.8, label=r"Input ")
ax.loglog(k_h, Pk, color="C5", lw=2, label=r"DISCO-DJ")

ax.set_ylabel(r"$P_{cb}(k)\;[(h^{-1}{\rm Mpc})^3]$")
ax.set_ylim(1e0, 4e4)
ax.legend()
ax.grid(True, which="both", alpha=0.18)
ax.set_title(rf"$\sum m_\nu = 0\,\mathrm{{eV}}$",fontsize=14)

axr.semilogx(k_h, abs(Pk/pk_nl_ref)-1 , color="C5", lw=1.8, label="DISCO-DJ/BACCO nonlinear")

axr.set_xlabel(r"$k\;[h\,{\rm Mpc}^{-1}]$")
axr.set_ylabel("deviation")
axr.set_xlim(k_h.min(), k_h.max())
axr.set_ylim(-0.5,0.5)
axr.legend()
axr.grid(True, which="both", alpha=0.18)

# %%

fig, (ax, axr) = plt.subplots(
    2,
    1,
    figsize=(7, 5),
    sharex=True,
    constrained_layout=True,
    gridspec_kw={"height_ratios": [2, 1]},
)
ax.loglog(k_hmpc, P_Nm_cb_h_nu, color="C1", lw=1.8,  label=r"CLASS NM ")
ax.loglog(k_h, pk_lin, color="C2", lw=1.8, ls=":", label=r"BACCO linear ")
ax.loglog(k_h, pk_nl, color="C3", lw=2.1, label=r"BACCO nonlinear ")
ax.loglog(k_pre_nu_nl, Pk_pre_nu_nl, color="C4", lw=1.8, label=r"Input")
ax.loglog(k_h,Pk_nu, color="C5", lw=2, label=r"DISCO-DJ ")

ax.set_ylabel(r"$P_{cb}(k)\;[(h^{-1}{\rm Mpc})^3]$")
ax.set_ylim(1e0, 4e4)
ax.legend()
ax.grid(True, which="both", alpha=0.18)
ax.set_title(rf"$\sum m_\nu = {sum_mnu}\,\mathrm{{eV}}$",fontsize=14)

axr.semilogx(k_h, abs(Pk_nu/pk_nl)-1, color="C5", lw=1.8, label="DISCO-DJ/BACCO nonlinear")

axr.set_xlabel(r"$k\;[h\,{\rm Mpc}^{-1}]$")
axr.set_ylabel("deviation")
axr.set_xlim(k_h.min(), k_h.max())
axr.set_ylim(-0.5,0.5)
axr.legend()
axr.grid(True, which="both", alpha=0.18)

# %%

fig, (ax, axr) = plt.subplots(
    2,
    1,
    figsize=(7, 5),
    sharex=True,
    constrained_layout=True,
    gridspec_kw={"height_ratios": [2, 1]},
)

ratio_class = P_Nm_cb_h_nu / P_Nm_cb_h
ratio_bacco    = pk_lin / pk_ref
ratio_bacco_nl = pk_nl  / pk_nl_ref
ratio_dj       = Pk_nu  / Pk

ax.semilogx(k_hmpc,ratio_class,color="C1",lw=1.8,label="CLASS NM",)
ax.semilogx(k_h,ratio_bacco,color="C2",lw=1.8,ls=":",label="BACCO linear",)
ax.semilogx(k_h,ratio_bacco_nl,color="C3",lw=1.8,label="BACCO nonlinear",)
ax.semilogx(k_h,ratio_dj,color="C5", linestyle='-.',lw=2,label="DISCO-DJ",)

ax.set_xscale("log")
ax.set_xlim(k_h.min(), k_h.max())
ax.set_xlabel(r"$k\;[h\,{\rm Mpc}^{-1}]$")
ax.set_ylabel(
    r"$P_{cb}(m_\nu) / P_{cb}(m_\nu=0)$"
)

ax.set_title(rf"Suppression due to $\sum m_\nu = {sum_mnu}$")
ax.legend(loc=3)
ax.grid(True, which="both", alpha=0.18)

axr.semilogx(k_h, abs(ratio_dj/ratio_bacco_nl-1), color="C5", lw=1.8, label="DISCO-DJ/BACCO nonlinear")

axr.set_xlabel(r"$k\;[h\,{\rm Mpc}^{-1}]$")
axr.set_ylabel("deviation")
axr.set_xlim(k_h.min(), k_h.max())
#axr.set_ylim(0.97,1.03)
axr.set_yscale("log")
axr.legend(prop={'size': 10},loc=2)
axr.grid(True, which="both", alpha=0.18)

