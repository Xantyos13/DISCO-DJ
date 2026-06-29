import jax.numpy as jnp
import jax
from scipy.optimize import fsolve

### Conversion factors
conKeV = 1.160451812e4  # conversion K/eV : 
conv_ev3_to_cm3 = (5.0677307e4)**3  # conversion eV3 / cm3

### Others ### 
Tcmb = 2.725  #  CMB temp (K)
Tnu0 = 0.71611*Tcmb # CNB temp (K) 
Tnu0_eV = Tnu0/conKeV # CNB temps (eV)
N_massive_nu = 3 # Number of massive neutrinos
g = 2 # DoF

def get_masses(delta_m_squared_atm, delta_m_squared_sol, sum_masses, hierarchy):
    """ 
        a function returning the three masses given the Delta m^2, the total mass, and the hierarchy (e.g. 'IN' or 'IH')
        taken from a piece of MontePython written by Thejs Brinckmann
    """
    # any string containing letter 'n' will be considered as refering to normal hierarchy
    if 'n' in hierarchy.lower():
        # Normal hierarchy massive neutrinos. Calculates the individual
        # neutrino masses from M_tot_NH and deletes M_tot_NH
        #delta_m_squared_atm=2.45e-3
        #delta_m_squared_sol=7.50e-5
        m1_func = lambda m1, M_tot, d_m_sq_atm, d_m_sq_sol: M_tot**2. + 0.5*d_m_sq_sol - d_m_sq_atm + m1**2. - 2.*M_tot*m1 - 2.*M_tot*(d_m_sq_sol+m1**2.)**0.5 + 2.*m1*(d_m_sq_sol+m1**2.)**0.5
        m1,opt_output,success,output_message = fsolve(m1_func,sum_masses/3.,(sum_masses,delta_m_squared_atm,delta_m_squared_sol),full_output=True)
        m1 = m1[0]
        m2 = (delta_m_squared_sol + m1**2.)**0.5
        m3 = (delta_m_squared_atm + 0.5*(m2**2. + m1**2.))**0.5
        return m1,m2,m3
    else:
        # Inverted hierarchy massive neutrinos. Calculates the individual
        # neutrino masses from M_tot_IH and deletes M_tot_IH
        #delta_m_squared_atm=-2.45e-3
        #delta_m_squared_sol=7.50e-5
        delta_m_squared_atm = -delta_m_squared_atm
        m1_func = lambda m1, M_tot, d_m_sq_atm, d_m_sq_sol: M_tot**2. + 0.5*d_m_sq_sol - d_m_sq_atm + m1**2. - 2.*M_tot*m1 - 2.*M_tot*(d_m_sq_sol+m1**2.)**0.5 + 2.*m1*(d_m_sq_sol+m1**2.)**0.5
        m1,opt_output,success,output_message = fsolve(m1_func,sum_masses/3.,(sum_masses,delta_m_squared_atm,delta_m_squared_sol),full_output=True)
        m1 = m1[0]
        m2 = (delta_m_squared_sol + m1**2.)**0.5
        m3 = (delta_m_squared_atm + 0.5*(m2**2. + m1**2.))**0.5
        return m1,m2,m3

################## BACKGROUND ################## 

def generalized_gauss_laguerre_weights(n, alpha):
    """
    Compute nodes and weights for n-point generalized Gauss-Laguerre quadrature,
    which approximates integrals of the form
        ∫₀∞ x^α f(x) e^(-x) dx.
    
    Parameters:
        n : int
            Number of quadrature points.
        alpha : float
            The parameter in the weight function x^α e^(-x).
    
    Returns:
        nodes : ndarray
            The quadrature nodes (abscissae).
        weights : ndarray
            The quadrature weights.
    """
    # Indices i = 1, 2, ..., n.
    i = jnp.arange(1, n+1)
    
    # Diagonal entries: a_i = 2i - 1 + alpha.
    a = 2*i - 1 + alpha
    
    # Off-diagonal entries for i = 1, ..., n-1: b_i = sqrt(i*(i+alpha))
    i_off = jnp.arange(1, n)
    b = jnp.sqrt(i_off * (i_off + alpha))
    
    # Construct the symmetric tridiagonal Jacobi matrix.
    J = jnp.diag(a) + jnp.diag(b, 1) + jnp.diag(b, -1)
    
    # Compute eigenvalues (nodes) and eigenvectors.
    nodes, V = jnp.linalg.eigh(J)
    
    # The weights are given by the square of the first component of the eigenvectors,
    # multiplied by the zeroth moment: Γ(α+1).
    if type(alpha) == int:
      weights = (V[0, :]**2) * jax.scipy.special.factorial(alpha)
    else:
      weights = (V[0, :]**2) * jax.scipy.special.gamma(alpha + 1)
    
    return nodes, weights


def get_neutrino_momentum_bins(  nqmax : int ) -> tuple[jax.Array, jax.Array]:
    """Get the momentum bins and integral kernel weights for neutrinos

    Args:
        nqmax (int): Number of momentum bins.

    Returns:
        jax.Array: q, w
    """
    # fermi_dirac_const = 7 * np.pi**4 / 120
    fermi_dirac_const = 5.682196976983475

    # nqmax = 3,4,5 are from high accuracy formulas from CAMB, higher values resort to modified Gauss-Laguerre, 
    # which is not pre-computed however
    if nqmax == 3:
        q = jnp.array([0.913201, 3.37517, 7.79184])
        dlfdlq = -q/(1+jnp.exp(-q))
        w = jnp.array([0.0687359, 3.31435, 2.29911]) / (-0.25*dlfdlq)
    elif nqmax == 4:
        q = jnp.array([0.7, 2.62814, 5.90428, 12.0])
        dlfdlq = -q/(1+jnp.exp(-q))
        w = jnp.array([0.0200251, 1.84539, 3.52736, 0.289427]) / (-0.25*dlfdlq)
    elif nqmax == 5:
        q = jnp.array([0.583165, 2.0, 4.0, 7.26582, 13.0])
        dlfdlq = -q/(1+jnp.exp(-q))
        w = jnp.array([0.0081201, 0.689407, 2.8063, 2.05156, 0.12681]) / (-0.25*dlfdlq)
    else:
        alpha = 1
        q, w = generalized_gauss_laguerre_weights( nqmax, alpha )
        w *= q**3 / (1 + jnp.exp(-q)) * q**-alpha
    
    return q, w


def nu_background( a : float, amnu: float, nq : int = 8 ) -> tuple[float, float, float]:
    """ computes the neutrino density and pressure of one flavour of massive neutrinos
        in units of the mean density of one flavour of massless neutrinos

    Args:
        a (float): scale factor
        amnu (float): neutrino mass in units of neutrino temperature (m_nu*c**2/(k_B*T_nu0).
        nq (int, optional): number of integration points. Defaults to 8.

    Returns:
        tuple[float, float, float]: rho_nu/rho_nu0, p_nu/p_nu0, pp_nu/pp_nu0
    """
    
    # q is the comoving momentum in units of k_B*T_nu0/c.
    v    = lambda q: 1 / jnp.sqrt(1 + (a * amnu / q)**2)   # = (1/aq) / sqrt(1+1/aq**2)
    
    q, w = get_neutrino_momentum_bins( nq )
    rhonu = jnp.dot( w, 1. / v(q) )
    pnu = jnp.dot( w, v(q) / 3 )
    ppnu = jnp.dot( w, v(q)**3 / 3 )
    
    return rhonu, pnu, ppnu


################## PERTURBATIVE ################## 


def nu_perturb( a : float, amnu: float, psi0: jax.Array, psi1 : jax.Array, psi2 : jax.Array, nqmax : int ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """ Compute the perturbations of density, energy flux, pressure, and
        shear stress of one flavor of massive neutrinos, in units of the mean
        density of one flavor of massless neutrinos, by integrating over 
        momentum.

    Args:
        a (float): scale factor
        amnu (float): neutrino mass in units of neutrino temperature (m_nu*c**2/(k_B*T_nu0).
        psi0 (jax.Array): l=0 neutrino perturbations for all momentum bins
        psi1 (jax.Array): l=1 neutrino perturbations for all momentum bins
        psi2 (jax.Array): l=2 neutrino perturbations for all momentum bins
        nq (int, optional): _description_. Defaults to 1000.
        qmax (float, optional): _description_. Defaults to 30..

    Returns:
        _type_: drhonu, dpnu, fnu, shearnu
    """
    
    q, w = get_neutrino_momentum_bins( nqmax )
    aq = a * amnu / q
    v = 1 / jnp.sqrt(1 + aq**2)

    drhonu = jnp.sum(w * psi0 / v)
    dpnu = jnp.sum(w * psi0 * v) / 3
    fnu = jnp.sum(w * psi1) 
    shearnu = jnp.sum(w * psi2 * v) * 2 / 3

    return drhonu, dpnu, fnu, shearnu

def nu_perturb_prime( a : float, amnu : float, aprimeoa : float, psi0: jax.Array, psi2: jax.Array, psi0prime : jax.Array, psi2prime : jax.Array, nqmax : int ) -> tuple[jax.Array, jax.Array]:
    """ Compute the time derivative of the mean density in massive neutrinos 
          and the shear perturbation.

    Args:
        a (float): scale factor
        aprimeoa (float): conformal Hubble rate
        amnu (float): neutrino mass in units of neutrino temperature (m_nu*c**2/(k_B*T_nu0).
        psi0 (jax.Array): l=0 neutrino perturbations for all momentum bins
        psi2 (jax.Array): l=2 neutrino perturbations for all momentum bins
        psi0prime (jax.Array): time derivative of l=0 neutrino perturbations for all momentum bins
        psi2prime (jax.Array): time derivative of l=2 neutrino perturbations for all momentum bins
        nqmax (int): number of momentum bins

    Returns:
        _type_: rho_nu_prime, shear_nu_prime
    """
    
    q, w = get_neutrino_momentum_bins( nqmax )
    aq = a * amnu / q
    aqprime = aprimeoa * aq
    v = 1 / jnp.sqrt(1 + aq**2)
    vprime = -aq*aqprime / (1+aq**2)**1.5

    # TODO: DOUBLE CHECK THESE:
    rho_nu_prime   = jnp.sum(w * (psi0prime / v - psi0 / v**2 * vprime))
    shear_nu_prime = jnp.sum(w * (psi2prime*v + psi2*vprime)) * 2 / 3

    return rho_nu_prime, shear_nu_prime


# @partial(jax.jit, static_argnames=('lmaxg', 'lmaxgp', 'lmaxr', 'lmaxnu', 'nqmax'))
def model_synchronous(*, tau, y, param, kmode, lmaxg, lmaxgp, lmaxr, lmaxnu, nqmax ):     
    """Solve the synchronous gauge perturbation equations for a single mode.

    Parameters
    ----------
    tau : float
        conformal time
    yin : array_like
        input vector of perturbations
    param : array_like
        dictionary of parameters and interpolated background functions
    kmode : float
        wavenumber of modef
    lmaxg : int
        maximum photon temperature hierarchy multipole
    lmaxgp : int
        maximum photon polarization hierarchy multipole
    lmaxr : int
        maximum massless neutrino hierarchy multipole
    lmaxnu : int
        maximum neutrino hierarchy multipole
    nqmax : int
        maximum number of momentum bins for massive neutrinos

    Returns
    -------
    f : array_like
        RHS of perturbation equations
    """
    Omegac = param['Omegam'] - param['Omegab']

    iq0 = 10 + lmaxg + lmaxgp + lmaxr
    iq1 = iq0 + nqmax
    iq2 = iq1 + nqmax
    iq3 = iq2 + nqmax
    iq4 = iq3 + nqmax

    # y = jnp.copy(yin)
    f = jnp.zeros_like( y )

    #TODO: add curvature
    # ... curvature
    K = 0
    
    # def cotKgen_zero_curv():
    #     return 1.0/(kmode*tau)
    # def cotKgen_pos_curv():
    #     return jnp.sqrt(K)/kmode/jnp.tan(jnp.sqrt(K)*tau)
    # def cotKgen_neg_curv():
    #     return jnp.sqrt(-K)/kmode/jnp.tanh(jnp.sqrt(-K)*tau)
    
    # cotKgen = jax.lax.switch(int(1+jax.lax.sign(K)), [cotKgen_neg_curv, cotKgen_zero_curv, cotKgen_pos_curv])
    s2_squared = 1.-3.*K/kmode**2
    s_l2 = 1.0
    s_l3 = 1.0

    # ... metric
    a = y[0]
    loga = jnp.log(a)

    #ahprime = y[1]
    eta = y[2]

    # ... cdm
    deltac = y[3]
    thetac = y[4]

    # ... baryons
    deltab = y[5]
    thetab = y[6]

    # ... photons
    deltag = y[7]
    thetag = y[8]
    shearg = y[9] / 2.0

    # ... massless neutrinos
    deltar = y[ 9 + lmaxg + lmaxgp]
    thetar = y[10 + lmaxg + lmaxgp]
    shearr = y[11 + lmaxg + lmaxgp] / 2.0

    # ... quintessence field
    deltaq = y[-2]
    thetaq = y[-1]

    # ... evaluate thermodynamics
    # tempb   = param['tempba_of_tau_spline'].evaluate( tau ) / a
    # xeprime = param['xe_of_tau_spline'].derivative( tau )
    # cs2     = param['cs2a_of_tau_spline'].evaluate( tau ) / a
    # xe      = param['xe_of_tau_spline'].evaluate( tau )

    # Use pre-composed splines for direct log(a) lookup (performance optimization)
    cs2     = param['cs2a_of_loga_spline'].evaluate( loga ) / a
    xe      = param['xe_of_loga_spline'].evaluate( loga )
    
    # ... Photon mass density over baryon mass density
    photbar = param['grhog'] / (param['grhom'] * param['Omegab'] * a)
    pb43 = 4.0 / 3.0 * photbar

    # massive neutrinos
    rhonu = jnp.exp(param['logrhonu_of_loga_spline'].evaluate(loga))
    # pnu = jnp.exp(param['logpnu_of_loga_spline'].evaluate( jnp.log(a) ) )

    # ... quintessence
    cs2_Q              = param['cs2_DE'] 
    w_Q                = param['w_DE_0'] + param['w_DE_a'] * (1.0 - a) 
    rho_Q              = a**(-3*(1+param['w_DE_0']+param['w_DE_a'])) * jnp.exp(3*(a-1)*param['w_DE_a'])
    rho_plus_p_theta_Q = (1+w_Q) * rho_Q * param['grhom'] * param['OmegaDE'] * thetaq * a**2
    
    # ... homogeneous background
    # grho = (
    #     param['grhom'] * param['Omegam'] / a
    #     + (param['grhog'] + param['grhor'] * (param['Neff'] + param['Nmnu'] * rhonu)) / a**2
    #     + param['grhom'] * param['OmegaDE'] * rho_Q * a**2
    #     + param['grhom'] * param['Omegak']
    # )

    # gpres = (
    #     (param['grhog'] + param['grhor'] * param['Neff']) / 3.0 + param['grhor'] * param['Nmnu'] * pnu
    # ) / a**2 + w_Q * param['grhom'] * param['OmegaDE'] * rho_Q * a**2

    # ... compute expansion rate
    aprimeoa = get_aprimeoa( param=param, aexp=a )
    # aprimeoa = jnp.sqrt(grho / 3.0)                # Friedmann I
    # aprimeprimeoa = 0.5 * (aprimeoa**2 - gpres)    # Friedmann II

    # quintessence EOS time derivatives
    w_Q_prime = -param['w_DE_a'] * aprimeoa * a
    ca2_Q     = w_Q - w_Q_prime / 3 / ((1+w_Q)+1e-6) / aprimeoa

    # ... Thomson opacity coefficient
    akthom = 2.3038921003709498e-9 * (1.0 - param['YHe']) * param['Omegab'] * param['H0']**2

    # ... Thomson opacity
    opac    = xe * akthom / a**2
    #tauc    = 1. / opac
    #taucprime = tauc * (2*aprimeoa - xeprime/xe)
    #F       = tauc / (1+pb43) #CLASS perturbations.c:10072
    #Fprime  = taucprime/(1+pb43) + tauc*pb43*aprimeoa/(1+pb43)**2 #CLASS perturbations.c:10074

    
    # ... background scale factor evolution
    f = f.at[0].set( aprimeoa * a )
    
    # ... evaluate metric perturbations
    drhonu, dpnu, fnu, shearnu = nu_perturb( a, param['amnu'], y[iq0:iq1], y[iq1:iq2], y[iq2:iq3], nqmax=nqmax )

    dgrho = (
        param['grhom'] * (Omegac * deltac + param['Omegab'] * deltab) / a
        + (param['grhog'] * deltag + param['grhor'] * (param['Neff'] * deltar + param['Nmnu'] * drhonu)) / a**2
        + param['grhom'] * param['OmegaDE'] * deltaq * rho_Q * a**2
    )
    dgpres = (
        (param['grhog'] * deltag + param['grhor'] * param['Neff'] * deltar) / a**2 / 3.0 
        + param['grhor'] * param['Nmnu'] * dpnu / a**2 
        + (cs2_Q * param['grhom'] * param['OmegaDE'] * deltaq * rho_Q * a**2 + (cs2_Q-ca2_Q)*(3*aprimeoa * rho_plus_p_theta_Q / kmode**2))
    )
    dgtheta = (
        param['grhom'] * (Omegac * thetac + param['Omegab'] * thetab) / a
        + 4.0 / 3.0 * (param['grhog'] * thetag + param['Neff'] * param['grhor'] * thetar) / a**2
        + param['Nmnu'] * param['grhor'] * kmode * fnu / a**2
        + rho_plus_p_theta_Q
    )
    dgshear = (
        4.0 / 3.0 * (param['grhog'] * shearg + param['Neff'] * param['grhor'] * shearr) / a**2
        + param['Nmnu'] * param['grhor'] * shearnu / a**2
    )

    dahprimedtau = -(dgrho + 3.0 * dgpres) * a
    
    f = f.at[1].set( dahprimedtau )

    # ... force energy conservation
    hprime = (2.0 * kmode**2 * eta + dgrho) / aprimeoa

    etaprime = 0.5 * dgtheta / kmode**2
    alpha  = (hprime + 6.*etaprime)/2./kmode**2
    f = f.at[2].set( etaprime )
    
    # alphaprime = -3*dgshear/(2*kmode**2) + eta - 2*aprimeoa*alpha
    # alphaprime -=  9/2 * a**2/kmode**2 * 4/3 * 16/45/opac * (thetag+kmode**2*alpha) * param['grhog']

    # ... cdm equations of motion, MB95 eq. (42)
    deltacprime = -thetac - 0.5 * hprime
    f = f.at[3].set( deltacprime )
    thetacprime = -aprimeoa * thetac  # thetac = 0 in synchronous gauge!
    f = f.at[4].set( thetacprime )

    idxb = 5
    # --- baryon equations of motion, MB95 eqs. (66) ---------------------------------------------
    # ... baryon density, BLT11 eq. (2.1a)
    deltabprime = -thetab - 0.5 * hprime
    f = f.at[idxb+0].set( deltabprime )
    # ... baryon velocity, BLT11 eq. (2.1b)
    thetabprime = -aprimeoa * thetab + kmode**2 * cs2 * deltab \
                + pb43 * opac * (thetag - thetab)
    f = f.at[idxb+1].set( thetabprime )

    # --- photon equations of motion, MB95 eqs. (63) ---------------------------------------------
    idxg  = 7
    idxgp = 7 + (lmaxg+1)
    # ... polarization term
    polter = y[idxg+2] + y[idxgp+0] + y[idxgp+2]
    # ... photon density, BLT11 eq. (2.4a)
    deltagprime = 4.0 / 3.0 * (-thetag - 0.5 * hprime)
    f = f.at[idxg+0].set( deltagprime )
    # ... photon velocity, BLT11 eq. (2.4b)
    thetagprime = kmode**2 * (0.25 * deltag - s2_squared * shearg) \
                - opac * (thetag - thetab)
    f = f.at[idxg+1].set( thetagprime )
    # ... photon shear, BLT11 eq. (2.4c)
    sheargprime = 8./15. * (thetag+kmode**2*alpha) -3/5*kmode*s_l3/s_l2*y[idxg+3] \
                - opac*(y[idxg+2]-0.1*s_l2*polter)
    f = f.at[idxg+2].set( sheargprime )

    #... photon temperature l>=3, BLT11 eq. (2.4d)
    ell  = jnp.arange(3, lmaxg )
    f = f.at[idxg+ell].set( kmode  / (2 * ell + 1) * (ell * y[idxg+ell-1] - (ell + 1) * y[idxg+ell+1]) - opac * y[idxg+ell] )
    # photon temperature hierarchy truncation, BLT11 eq. (2.5)
    f = f.at[idxg+lmaxg].set( kmode * y[idxg+lmaxg-1] - (lmaxg + 1) / tau * y[idxg+lmaxg] - opac * y[idxg+lmaxg] )

    #... polarization equations, BLT11 eq. (2.4e)
    ell  = jnp.arange(0, lmaxgp) # l=0...lmaxgp-1
    f = f.at[idxgp+ell].set( kmode  / (2 * ell + 1) * (ell * y[idxgp+ell-1] - (ell + 1) * y[idxgp+ell+1]) - opac * y[idxgp+ell] )
    f = f.at[idxgp+0].add( opac * polter / 2 )  # photon polarization l=0
    f = f.at[idxgp+2].add( opac * polter / 10 ) # photon polarization l=2
    
    # photon polarization hierarchy truncation
    f = f.at[idxgp+lmaxgp].set( kmode * y[idxgp+lmaxgp-1] - (lmaxgp + 1) / tau * y[idxgp+lmaxgp] - opac * y[idxgp+lmaxgp] )
    
    
    # --- Massless neutrino equations of motion -------------------------------------------------------
    idxr = 9 + lmaxg + lmaxgp
    deltarprime = 4.0 / 3.0 * (-thetar - 0.5 * hprime)
    f = f.at[idxr+0].set( deltarprime )
    thetarprime = kmode**2 * (0.25 * deltar - shearr)
    f = f.at[idxr+1].set( thetarprime )
    shearrprime = 8./15. * (thetar + kmode**2 * alpha) - 0.6 * kmode * y[idxr+3]
    f = f.at[idxr+2].set( shearrprime )
    ell = jnp.arange(3, lmaxr)
    f = f.at[idxr+ell].set( kmode / (2 * ell + 1) * (ell * y[idxr+ell-1] - (ell + 1) * y[idxr+ell+1]) )
    
    # ... truncate moment expansion
    f = f.at[idxr+lmaxr].set( kmode * y[idxr+lmaxr-1] - (lmaxr + 1) / tau * y[idxr+lmaxr] )

    # --- Massive neutrino equations of motion --------------------------------------------------------
    # q = jnp.arange(1, nqmax + 1) - 0.5  # so dq == 1 # if not using CAMB approx
    q, _ = get_neutrino_momentum_bins( nqmax )
    aq = a * param['amnu'] / q
    v = 1 / jnp.sqrt(1 + aq**2)
    dlfdlq = -q / (1.0 + jnp.exp(-q))  # derivative of the Fermi-Dirac distribution

    f = f.at[iq0 : iq1].set(
        -kmode * v * y[iq1 : iq2] + hprime* dlfdlq / 6.0 
    )
    f = f.at[iq1 : iq2].set(
        kmode * v * (y[iq0 : iq1] - 2.0 * y[iq2 : iq3]) / 3.0
    )
    f = f.at[iq2 : iq3].set(
        kmode * v * (2 * y[iq1 : iq2] - 3 * y[iq3 : iq4]) / 5.0 - (hprime / 15 + 2 / 5 * etaprime) * dlfdlq
    )

    ell = jnp.arange(3, lmaxnu)
    vv = jnp.tile(v, lmaxnu - 3)
    denl = jnp.repeat( 2*ell+1, nqmax )

    f = f.at[iq0 + 3 * nqmax : iq0 + lmaxnu * nqmax].set(
        kmode * vv / denl * (
            jnp.repeat( ell, nqmax ) * y[iq0 + 2*nqmax : iq0 + (lmaxnu-1)*nqmax] 
            - jnp.repeat( ell+1, nqmax ) * y[iq0 + 4*nqmax : iq0 + (lmaxnu+1)*nqmax]
        ) 
    )

    # Truncate moment expansion.
    f = f.at[-nqmax-2 :-2].set(
        kmode * v * y[-2 * nqmax-2 : -nqmax-2] - (lmaxnu + 1) / tau * y[-nqmax-2 :-2]
    )

    # ---- Quintessence equations of motion -----------------------------------------------------------
    # ... Ballesteros & Lesgourgues (2010, BL10), arXiv:1004.5509
    f = f.at[-2].set( # BL10, eq. (3.5)
        -(1+w_Q) *(thetaq + 0.5 * hprime) - 3*(cs2_Q - w_Q) * aprimeoa * deltaq - 9*(1+w_Q)*(cs2_Q-ca2_Q)*aprimeoa**2/kmode**2 * thetaq
    )
    f = f.at[-1].set( # BL10, eq. (3.6)
        -(1-3*cs2_Q)*aprimeoa*thetaq + cs2_Q/(1+w_Q) * kmode**2 * deltaq
    )

    return f.flatten()


def convert_to_output_variables(*, y, param, kmode, lmaxg, lmaxgp, lmaxr, lmaxnu, nqmax ):
    """Convert the synchronous gauge perturbations to the output fields.

    Parameters
    ----------
    y : array_like
        input vector of perturbations
    param : dict
        dictionary of parameters and interpolated background functions
    kmode : float
        wavenumber of mode [1/Mpc]
    lmaxg : int
        maximum photon Boltmann hierarchy moment
    lmaxgp : int
        maximum photon polarization hierarchy moment
    lmaxr : int
        maximum massless neutrino hierarchy moment
    lmaxnu : int
        maximum massive neutrino hierarchy moment
    nqmax : int
        number of momentum bins for massive neutrinos

    Returns
    -------
    yout : array_like
        output vector of perturbations:
            eta, etaprime, hprime, alpha,       # 0-3
            deltam,  thetam / (aH),             # 4-5
            deltabc, thetabc / (aH),            # 6-7
            deltac,  thetac / (aH),             # 8-9
            deltab,  thetab / (aH),             # 10-11
            deltag,  thetag / (aH),             # 12-13
            deltar,  thetar / (aH),             # 14-15
            deltanu, thetanu / (aH),            # 16-17
            deltaq,  thetaq / (aH),             # 18-19
    where aH = \mathcal{H} = a' / a, which is the conformal Hubble rate.
    """

    Omegac = param['Omegam'] - param['Omegab']

    iq0 = 10 + lmaxg + lmaxgp + lmaxr
    iq1 = iq0 + nqmax
    iq2 = iq1 + nqmax
    iq3 = iq2 + nqmax

    a = y[0]
    eta = y[2]

    # ... cdm
    deltac = y[3]
    thetac = y[4]

    # ... baryons
    deltab = y[5]
    thetab = y[6]

    # ... photons
    deltag = y[7]
    thetag = y[8]

    # ... massless neutrinos
    deltar = y[ 9 + lmaxg + lmaxgp]
    thetar = y[10 + lmaxg + lmaxgp]
  
    rhonu = jnp.exp(param['logrhonu_of_loga_spline'].evaluate(jnp.log(a)))
    pnu = jnp.exp(param['logpnu_of_loga_spline'].evaluate( jnp.log(a) ) )
    rho_plus_p = rhonu + pnu

    drhonu, _, fnu, _ = nu_perturb( a, param['amnu'], y[iq0:iq1], y[iq1:iq2], y[iq2:iq3], nqmax=nqmax )
    deltanu = drhonu / rhonu
    thetanu = kmode * fnu / rho_plus_p

    # ... quintessence field
    deltaq    = y[-2]
    thetaq    = y[-1]
    w_Q       = param['w_DE_0'] + param['w_DE_a'] * (1.0 - a)
    rho_Q     = a**(-3*(1+param['w_DE_0']+param['w_DE_a'])) * jnp.exp(3*(a-1)*param['w_DE_a'])
    rho_plus_p_theta_Q = (1+w_Q) * rho_Q * param['grhom'] * param['OmegaDE'] * thetaq * a**2


    # ... background
    grho = (
        param['grhom'] * param['Omegam'] / a
        + (param['grhog'] + param['grhor'] * (param['Neff'] + param['Nmnu'] * rhonu)) / a**2
        + param['grhom'] * param['OmegaDE'] * rho_Q * a**2
        + param['grhom'] * param['Omegak']
    )

    gpres = (
        (param['grhog'] + param['grhor'] * param['Neff']) / 3.0 + param['grhor'] * param['Nmnu'] * pnu
    ) / a**2 + w_Q * param['grhom'] * param['OmegaDE'] * rho_Q * a**2
    
    # ... compute expansion rate
    aprimeoa = jnp.sqrt(grho / 3.0)                # Friedmann I
    
    # ... metric perturbations
    dgrho = (
        param['grhom'] * (Omegac * deltac + param['Omegab'] * deltab) / a
        + (param['grhog'] * deltag + param['grhor'] * (param['Neff'] * deltar + param['Nmnu'] * drhonu)) / a**2
        + param['grhom'] * param['OmegaDE'] * deltaq * rho_Q * a**2
    )
    dgtheta = (
        param['grhom'] * (Omegac * thetac + param['Omegab'] * thetab) / a
        + 4.0 / 3.0 * (param['grhog'] * thetag + param['Neff'] * param['grhor'] * thetar) / a**2
        + param['Nmnu'] * param['grhor'] * kmode * fnu / a**2
        + rho_plus_p_theta_Q
    )
    
    hprime = (2.0 * kmode**2 * eta + dgrho) / aprimeoa
    etaprime = 0.5 * dgtheta / kmode**2
    alpha  = (hprime + 6.*etaprime)/2./kmode**2


    # total matter perturbations
    deltam = (
        ( param['grhom'] * (Omegac * deltac + param['Omegab'] * deltab) / a
        + (param['grhor'] * param['Nmnu'] * drhonu) / a**2) / (param['grhom'] * param['Omegam'] / a
        + (param['grhor'] * param['Nmnu'] * rhonu)/ a**2 )
    )
    thetam = (
        (param['grhom'] * (Omegac * thetac + param['Omegab'] * thetab) / a + param['Nmnu'] * param['grhor'] * kmode * fnu / a**2) 
        / (3.0 * (param['grhom'] * param['Omegam'] / a + param['grhor'] * param['Nmnu'] * rhonu / a**2 ))
    )

    deltabc = (param['grhom'] * (Omegac * deltac + param['Omegab'] * deltab)/ a) \
        / (param['grhom'] * param['Omegam'] / a)
    thetabc = (param['grhom'] * (Omegac * thetac + param['Omegab'] * thetab) / a) \
        / (3.0 * (param['grhom'] * param['Omegam'] / a) / a**2)
    
    #... gauge trafo from comoving (MB95 eq. 27b)
    thetam   += alpha * kmode**2
    thetabc  += alpha * kmode**2

    ##################################################################################################################

    # store fields of interest
    yout = jnp.array([
        eta, etaprime, hprime, alpha,       # 0-3
        deltam,  thetam  / aprimeoa,        # 4-5
        deltabc, thetabc / aprimeoa,        # 6-7
        deltac,  thetac  / aprimeoa,        # 8-9
        deltab,  thetab  / aprimeoa,        # 10-11
        deltag,  thetag  / aprimeoa,        # 12-13
        deltar,  thetar  / aprimeoa,        # 14-15
        deltanu, thetanu / aprimeoa,        # 16-17
        deltaq,  thetaq  / aprimeoa,        # 18-19
    ])
            
    return yout

######################### WITH CLASS #########################

    @forbidden_for_derivative
    def deltanu_with_class(self):

        from classy import Class
        import numpy as np

        class_params = {
            "output": "mPk,dTk",
            "h": self.h,
            "omega_b": self.Omega_b * self.h**2,
            "omega_cdm": self.Omega_c * self.h**2,
            "A_s": 2.089e-9,
            "n_s": self.n_s,
            "tau_reio": 0.0952,
            "N_ncdm": 3,
            "m_ncdm": f"{self.mnu[0]},{self.mnu[1]},{self.mnu[2]}",
            "N_ur": 0,
            "P_k_max_h/Mpc": 10.,
            "z_max_pk": 50.,
            "T_cmb": 2.7255,
            "YHe": 0.24,
        }

        cosmo = Class()
        cosmo.set(class_params)
        cosmo.compute()

        a_values = np.linspace(0.02, 1.0, 50)

        # on récupère la grille k interne de CLASS
        tr0 = cosmo.get_transfer(z=0.)
        k_h = tr0['k (h/Mpc)']

        nk = len(k_h)
        na = len(a_values)

        delta_nu = np.zeros((nk, na, 3))

        for ia, a in enumerate(a_values):

            z = 1./a - 1.

            tr = cosmo.get_transfer(z=z)

            delta_nu[:, ia, 0] = tr['d_ncdm[0]']
            delta_nu[:, ia, 1] = tr['d_ncdm[1]']
            delta_nu[:, ia, 2] = tr['d_ncdm[2]']

        cosmo.struct_cleanup()
        cosmo.empty()

        return k_h, a_values, delta_nu



def get_nu_correction_k(self, k_vecs, deltanu_table, a_deltanu, a):
    """
    Compute
        correction(k,a) = sum_i Omega_nu_i(a)/Omega_m(a) * delta_nu_i(k,a)
    interpolated on the PM Fourier grid.

    Parameters
    ----------
    k_vecs : tuple/list
        Fourier vectors of the PM mesh.
    deltanu_table : array
        Shape (Nk, Na, 3)
    a_deltanu : array
        Scale factors corresponding to deltanu_table.
    a : float
        Scale factor.

    Returns
    -------
    correction : array
        Shape identical to the PM Fourier mesh.
    """

    # Fourier-space |k|
    k_mag = jnp.sqrt(
        jnp.sum(
            jnp.stack(
                [k_vecs[d]**2 for d in range(self.dim)],
                axis=0
            ),
            axis=0
        )
    )

    k_mag_h = k_mag / self.h

    # nearest time slice
    a_idx = jnp.argmin(jnp.abs(a_deltanu - a))

    # Omega_m(a)
    Omega_m_a = self.Omega_m * a**(-3)

    correction = jnp.zeros_like(k_mag)

    for i in range(3):

        Omega_nu_i = self.Omega_nu(a, flavor=i+1)

        weight = Omega_nu_i / Omega_m_a

        delta_nu_i = deltanu_table[:, a_idx, i]

        delta_nu_interp = jnp.interp(
            k_mag_h.flatten(),
            self.k_deltanu,
            delta_nu_i
        ).reshape(k_mag.shape)

        correction += weight * delta_nu_interp

    return correction

