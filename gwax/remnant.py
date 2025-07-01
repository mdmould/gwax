# all credit to https://github.com/dgerosa/precession and references therein

import jax
import jax.numpy as jnp


def eval_eta(q):
    return q / (1 + q)**2

def eval_theta12(theta1, theta2, deltaphi):
    return jnp.arccos(
        jnp.sin(theta1) * jnp.sin(theta2) * jnp.cos(deltaphi)
        + jnp.cos(theta1) * jnp.cos(theta2)
    )

def angles_to_Lframe(theta1, theta2, deltaphi, r, q, chi1, chi2):
    L = r**0.5 * q / (1 + q)**2
    S1 = chi1 / (1 + q)**2
    S2 = chi2 * q**2 / (1 + q)**2

    Lx = 0
    Ly = 0
    Lz = L
    Lvec = jnp.array([Lx, Ly, Lz])

    S1x = S1 * jnp.sin(theta1)
    S1y = 0
    S1z = S1 * jnp.cos(theta1)
    S1vec = jnp.array([S1x, S1y, S1z])

    S2x = S2 * jnp.sin(theta2) * jnp.cos(deltaphi)
    S2y = S2 * jnp.sin(theta2) * jnp.sin(deltaphi)
    S2z = S2 * jnp.cos(theta2)
    S2vec = jnp.array([S2x, S2y, S2z])

    return Lvec, S1vec, S2vec


# dimensionless remnant mass, i.e., in units of total pre-merger binary mass
def remnant_mass(theta1, theta2, q, chi1, chi2):
    eta = eval_eta(q)

    chit_par =  ( chi2*q**2 * jnp.cos(theta2) + chi1*jnp.cos(theta1) ) / (1+q)**2

    #Final mass. Barausse Morozova Rezzolla 2012
    p0 = 0.04827
    p1 = 0.01707
    Z1 = 1 + (1-chit_par**2)**(1/3)* ((1+chit_par)**(1/3)+(1-chit_par)**(1/3))
    Z2 = (3* chit_par**2 + Z1**2)**(1/2)
    risco = 3 + Z2 - jnp.sign(chit_par) * ((3-Z1)*(3+Z1+2*Z2))**(1/2)
    Eisco = (1-2/(3*risco))**(1/2)
    #Radiated energy, in units of the initial total mass of the binary
    Erad = eta*(1-Eisco) + 4* eta**2 * (4*p0+16*p1*chit_par*(chit_par+1)+Eisco-1)
    Mfin = 1- Erad # Final mass

    return Mfin


def remnant_spin(theta1, theta2, deltaphi, q, chi1, chi2):
    eta = eval_eta(q)

    kfit = jnp.array( [[jnp.nan, 3.39221, 4.48865, -5.77101, -13.0459] ,
                      [35.1278, -72.9336, -86.0036, 93.7371, 200.975],
                      [-146.822, 387.184, 447.009, -467.383, -884.339],
                      [223.911, -648.502, -697.177, 753.738, 1166.89]])
    xifit = 0.474046

    # Calculate K00 from Eq 11
    kfit = kfit.at[0,0].set(4**2 * ( 0.68646 - jnp.sum( kfit[1:,0] /(4**(3+jnp.arange(kfit.shape[0]-1)))) - (3**0.5)/2))

    theta12 = eval_theta12(theta1, theta2, deltaphi)

    eps1 = 0.024
    eps2 = 0.024
    eps12 = 0
    theta1 = theta1 + eps1 * jnp.sin(theta1)
    theta2 = theta2 + eps2 * jnp.sin(theta2)
    theta12 = theta12 + eps12 * jnp.sin(theta12)

    # Eq. 14 - 15
    atot = ( chi1*jnp.cos(theta1) + chi2*jnp.cos(theta2)*q**2 ) / (1+q)**2
    aeff = atot + xifit*eta* ( chi1*jnp.cos(theta1) + chi2*jnp.cos(theta2) )

    # Eq. 2 - 6 evaluated at aeff, as specified in Eq. 11
    Z1= 1 + (1-(aeff**2))**(1/3) * ( (1+aeff)**(1/3) + (1-aeff)**(1/3) )
    Z2= ( (3*aeff**2) + (Z1**2) )**(1/2)
    risco= 3 + Z2 - jnp.sign(aeff) * ( (3-Z1)*(3+Z1+2*Z2) )**(1/2)
    Eisco=(1-2/(3*risco))**(1/2)
    Lisco = (2/(3*(3**(1/2)))) * ( 1 + 2*(3*risco - 2 )**(1/2) )

    # Eq. 13
    etatoi = eta**(1+jnp.arange(kfit.shape[0]))
    innersum = jnp.sum(kfit.T * etatoi,axis=1)
    aefftoj = aeff**(jnp.arange(kfit.shape[1]))
    sumell = jnp.sum(innersum  * aefftoj,axis=0)
    ell = jnp.abs( Lisco  - 2*atot*(Eisco-1)  + sumell )

    # Eq. 16
    chifin = (1/(1+q)**2) * ( chi1**2 + (chi2**2)*(q**4)  + 2*chi1*chi2*(q**2)*jnp.cos(theta12)
            + 2*(chi1*jnp.cos(theta1) + chi2*(q**2)*jnp.cos(theta2))*ell*q + ((ell*q)**2)  )**(1/2)

    return jnp.minimum(chifin,1)


def remnant_kick(bigTheta, theta1, theta2, deltaphi, q, chi1, chi2):
# kms=False, maxphase=False, superkick=True, hangupkick=True, crosskick=True, full_output=False):

    eta = eval_eta(q)

    Lvec, S1vec, S2vec = angles_to_Lframe(theta1, theta2, deltaphi, 1, q, chi1, chi2)
    hatL = Lvec / jnp.linalg.norm(Lvec)
    hatS1 = S1vec / jnp.linalg.norm(S1vec)
    hatS2 = S2vec / jnp.linalg.norm(S2vec)

    #More spin parameters.
    Delta = - 1/(1+q) * (q*chi2*hatS2 - chi1*hatS1)
    Delta_par = jnp.dot(Delta, hatL)
    Delta_perp = jnp.linalg.norm(jnp.cross(Delta, hatL))
    chit = 1/(1+q)**2 * (chi2*q**2*hatS2 + chi1*hatS1)
    chit_par = jnp.dot(chit, hatL)
    chit_perp = jnp.linalg.norm(jnp.cross(chit, hatL))

    #Coefficients are quoted in km/s
    #vm and vperp from Kesden at 2010a. vpar from Lousto Zlochower 2013
    zeta=jnp.radians(145)
    A=1.2e4
    B=-0.93
    H=6.9e3

    #Multiply by 0/1 boolean flags to select terms
    V11 = 3677.76
    VA = 2481.21
    VB = 1792.45
    VC = 1506.52
    C2 = 1140
    C3 = 2481

    # #maxkick
    # bigTheta=np.random.uniform(0, 2*np.pi,q.shape) * (not maxphase)

    vm = A * eta**2 * (1+B*eta) * (1-q)/(1+q)
    vperp = H * eta**2 * Delta_par
    vpar = 16*eta**2 * (Delta_perp * (V11 + 2*VA*chit_par + 4*VB*chit_par**2 + 8*VC*chit_par**3) + chit_perp * Delta_par * (2*C2 + 4*C3*chit_par)) * jnp.cos(bigTheta)
    kick = jnp.array([vm+vperp*jnp.cos(zeta),vperp*jnp.sin(zeta),vpar]).T

    # if not kms:
    #     kick = kick/299792.458 # speed of light in km/s

    vk = jnp.linalg.norm(kick)

    return vk
