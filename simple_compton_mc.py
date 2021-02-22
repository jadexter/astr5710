import numpy as np

# physical constants
kb=1.38e-16; c=3e10; h=6.63e-27; me=9.11e-28; e=4.8e-10
sigT=6.63e-25

#velmag=0.; bulkgam=0.; tstep=0.; Thetae=0.01; ne=1e4; xmin=1e-3; xmax=1e-3
#lxmin=np.log(xmin); lxmax=np.log(xmax)
n_samp=100; R=1e14; dt_frac=0.01; tstep=0.
n_time_steps=100

# option to choose parameters, otherwise set to default options
def init_parameters(velmag_in=2.8e10,Thetae_in=1e-2,ne_in=4e8,xmin_in=1e-3,xmax_in=1e3,n_samp_in=100000,R_in=1e14,n_time_steps_in=1000):
    global velmag,bulkgam,Thetae,ne,xmin,xmax,lxmin,lxmax,n_samp,R
    global dt_frac,n_time_steps,tstep,tstart

    tstart=0.
    
    # bulk velocity
    #velmag=2.9999e10
    velmag=velmag_in
    bulkgam=1./np.sqrt(1.-(velmag/c)**2.)

    # variable definitions in cgs
    Thetae=Thetae_in; ne=ne_in
    xmin=xmin_in; xmax=xmax_in
    lxmin=np.log(xmin); lxmax=np.log(xmax)
    n_samp=n_samp_in


    # region size in cm
    R = R_in
    
    # restrict a time step to a small fraction of the domain
    dt_frac=0.01

    # time step in s
    tstep=dt_frac*R/c

    # number of time steps
    n_time_steps = n_time_steps_in

    tausc=ne*R*sigT; y=max(tausc,tausc**2.)*max(4.*Thetae,4./3.*bulkgam**2.)
    print('tausc: ',tausc)
    print('compton y: ',y)

# bremsstrahlung routines
def J_brem(ne, Thetae):
    Te = Thetae*me*c*c/kb
    gff = 1.2;
    rel = (1. + 4.4e-10*Te)
    Jb = np.sqrt(2.*np.pi*kb*Te/(3.*me))
    Jb *= 2**5.*np.pi*e**6./(3*h*me*c**3.)
    Jb *= ne*ne*gff*rel
    return Jb

def jnu_brem(nu, ne, Thetae):
    Te = Thetae*me*c*c/kb
    gff=1.2
    rel = (1. + 4.4e-10*Te)
    x = h*nu/(kb*Te)
    efac = np.exp(-x)
    jv = 1./(4.*np.pi)*2**5.*np.pi*e**6./(3.*me*c**3.)
    jv *= np.sqrt(2.*np.pi/(3.*kb*me))
    jv *= Te**(-1./2.)*ne*ne
    jv *= efac*rel*gff

    return jv

def Jnu_brem(nu, ne, Thetae):
    return 4.*np.pi*jnu_brem(nu, ne, Thetae)

## NNU DIST ##
def nnu_brem(x,x_min):
    return x_min/x*np.exp(-x)

## ENU DIST ##
def enu_brem(x,x_max):
    return x*np.exp(-x)

# random samples from bremsstrahlung emissivity spaced in energy
def sample_nu_brem(x_min,x_max,Thetae,nphoton=1e4):
    nu=np.zeros(nphoton)
    lx_min=np.log(x_min); lx_max=np.log(x_max)
    n_needed = nphoton; n_samples = 0
    while np.any(nu==0.):
        u=np.exp(np.random.rand(n_needed)*(lx_max-lx_min)+lx_min)/x_max
#        z=nnu_brem(u*x_max,x_min)
        z=enu_brem(u*x_max,x_max)
        v=np.random.rand(n_needed)
        conv = (v <= z); nconv=np.sum(conv)
        nu[n_samples:n_samples+nconv]=u[conv]*x_max
        n_samples+=nconv; n_needed=nphoton-n_samples
#        print(n_samples,n_needed,np.any(nu==0.))
#        print(u[u*x_max > 1.]*x_max)
#        print(z[u*x_max > 1.])
    Te = Thetae*me*c*c/kb
    nu*=kb*Te/h
    return nu

# random, isotropic velocity
def isotropic_vel(n):
    mu=1.-2.*np.random.rand(n)
    phi=2.*np.pi*np.random.rand(n)
    smu=np.sqrt(1.-mu*mu)
    Dx=smu*np.cos(phi)
    Dy=smu*np.sin(phi)
    Dz=mu
    return Dx,Dy,Dz

# assumes Thetae << 1
def sample_maxwell(Thetae,n):
    vx=np.zeros(n); vy=np.zeros(n); vz=np.zeros(n)
    vx = np.random.randn(n)
    vz = np.random.randn(n)
    vy = np.random.randn(n)
    # multiply by expectation value
    Te = Thetae*me
    vx *= c*np.sqrt(Thetae)
    vy *= c*np.sqrt(Thetae)
    vz *= c*np.sqrt(Thetae)
    return vx,vy,vz

## INITIALIZATION PIECES

def init_photons(n_samp,xmin,xmax):
    global xph,yph,zph,rph,tph,tstart
    global R,ne,Thetae,nu,nu_input
    global Dx,Dy,Dz,Vol,Lum,Ep,E_input
    # random positions in xyz starting near the center
    tstart=0.
    rstart=R/10

    xph=rstart*(2.*np.random.rand(n_samp)-1.)
    yph=rstart*(2.*np.random.rand(n_samp)-1.)
    zph=rstart*(2.*np.random.rand(n_samp)-1.)

    rph=np.sqrt(xph*xph+yph*yph+zph*zph)
    thph=np.arccos(zph/rph)
    phph=np.arctan2(yph,xph)
    tph = rph*0.

    # isotropic directions for photons
    Dx,Dy,Dz = isotropic_vel(n_samp)

    # calculate total luminosity and so energy per photon packet
    Vol=4./3.*R**3.
    Lum = J_brem(ne, Thetae)*Vol

    # this is emitting over one time step, but arbitrary here unless we continue to emit
    Ep = Lum*tstep/n_samp+np.zeros(n_samp)
    E_input=Ep.copy()

    # frequency sampling

    # Bremsstrahlung
    nu=sample_nu_brem(xmin,xmax,Thetae,nphoton=n_samp)

    # narrow Gaussian of soft photons
    #nuc=1e14
    #nu=np.exp(np.random.randn(n_samp))*nuc
    nu_input=nu.copy()

# now we want to be able to takea time step
def step(tstart):
    global xph,yph,zph,rph,tph,Dx,Dy,Dz,Ep,nu,Thetae,ne,R
    E_save=[]; nu_save=[]
    # which photons are still here?
    active = rph < R
    n_active = np.sum(active)
    
    if np.sum(~active) > 0:
        E_save.append(Ep[~active])
        nu_save.append(nu[~active])
    
    if n_active==0: return tstart,E_save,nu_save
    
    # make sure we are only indexing to active photons
    xph=xph[active]
    yph=yph[active]
    zph=zph[active]
    rph=rph[active]
    tph = tph[active]
    Dx=Dx[active]; Dy=Dy[active]; Dz=Dz[active]
    Ep=Ep[active]
    nu=nu[active]
    
    print('active photons: ',np.sum(active))

    tstop=tstart+tstep
    # d_bn ???
    scatter=np.ones(n_active)
    # loop until all particles have finished step
    while(np.any(scatter)):
    
        # choose random optical depth until next interaction for each photon
        tau_r = -1.*np.log(1.-np.random.rand(n_active))
    
        # step size to interaction, for now Thomson cross section
        d_sc = tau_r/(ne*sigT)
    
        # distance to end of time step
        d_tm = (tstop-tph)*c
    
        # find out what happens
        scatter = d_sc <= d_tm
        take_step = d_tm < d_sc
    
        if np.sum(scatter) > 0:
            # move the photons
            xph[scatter] += d_sc[scatter]*Dx[scatter]
            yph[scatter] += d_sc[scatter]*Dy[scatter]
            zph[scatter] += d_sc[scatter]*Dz[scatter]
            # time at scatter
            tph[scatter] += d_sc[scatter]/c
            rph[scatter] = np.sqrt(xph[scatter]**2.+yph[scatter]**2.+zph[scatter]**2.)
        
            # scatter the photons
            compton_scatter(scatter)

        if np.sum(take_step) > 0:
            xph[take_step] += d_tm[take_step]*Dx[take_step]
            yph[take_step] += d_tm[take_step]*Dy[take_step]
            zph[take_step] += d_tm[take_step]*Dz[take_step]
            tph[take_step] += d_tm[take_step]/c
            rph[take_step] = np.sqrt(xph[take_step]**2.+yph[take_step]**2.+zph[take_step]**2.)
    
    return tstop,E_save,nu_save

def compton_scatter(indx):
    global Thetae,nu,Ep
    n_scatter=np.sum(indx); energy=Ep[indx]; nu0=nu[indx]
    Dx0=Dx[indx]; Dy0=Dy[indx]; Dz0=Dz[indx]
    # draw random velocities at interaction from Maxwell-Boltzmann dist
#    vx,vy,vz=sample_maxwell(Thetae,n_scatter)
    # set constant velocity
    vx,vy,vz=isotropic_vel(n_scatter)
    vx*=velmag
    vy*=velmag
    vz*=velmag
    
    # transform quantities into the comoving frame
    vdotD=vx*Dx0+vy*Dy0+vz*Dz0
    gamma_arr=1./np.sqrt(1.-(vx**2.+vy**2.+vz**2.)/c**2.)
    dshift_in = gamma_arr*(1.-vdotD/c)
    
    energy = energy*dshift_in; nu0=nu0*dshift_in

    # sample new direction by rejection method
    Er=np.zeros(n_scatter)-1.
    for i in range(n_scatter):
        stop=0
        Dxt=Dx0[i]; Dyt=Dy0[i]; Dzt=Dz0[i]; nut=nu0[i]
        energyt=energy[i]; gamma=gamma_arr[i]
#        print('scatter: ',i,n_scatter)
        while(stop==0):
            Dx_new,Dy_new,Dz_new = isotropic_vel(1)
            u=np.random.rand(1)
            cost = Dxt*Dx_new+Dyt*Dy_new+Dzt*Dz_new
            E_ratio = 1./(1.+h*nut/me/c/c*(1.-cost))
            diff_cs = 0.5*(E_ratio*E_ratio/(1/E_ratio+E_ratio-1+cost*cost))
            if u <= diff_cs:
#                Er[i]=E_ratio
                stop=1
            # why doesn't energy also get multiplied by Er?
        nut=nut*E_ratio; energyt=energyt*E_ratio
        vx[i]=-vx[i]; vy[i]=-vy[i]; vz[i]=-vz[i]
        vdp=vx[i]*Dx_new+vy[i]*Dy_new+vz[i]*Dz_new
        vd_out = gamma*(1.-vdp/c)
    
        Dxt = 1.0/vd_out*(Dx_new-gamma*vx[i]/c*(1-gamma*vdp/c/(gamma+1)))
        Dyt = 1.0/vd_out*(Dy_new - gamma*vy[i]/c*(1 - gamma*vdp/c/(gamma+1)))
        Dzt = 1.0/vd_out*(Dz_new - gamma*vz[i]/c*(1 - gamma*vdp/c/(gamma+1)))
        norm = np.sqrt(Dxt*Dxt + Dyt*Dyt + Dzt*Dzt)
#        print(vd_out,gamma,Dx_new)
        Dx0[i] = Dxt/norm
        Dy0[i] = Dyt/norm
        Dz0[i] = Dzt/norm
        
        # transform packet frequency and total energy
        nu0[i] = nut*vd_out
        energy[i] = energyt*vd_out

    Dx[indx] = Dx0
    Dy[indx] = Dy0
    Dz[indx] = Dz0

    # transformation of energy/wavelength into lab frame
    Ep[indx] = energy; nu[indx]= nu0

    if np.sum(nu[indx] < 0.) > 0:
        print('nult0: ',vd_out,gamma,dshift_in,Er)

# run a calculation
def run():
    global total_energy,total_nu,nu_input
    global tstart,n_time_steps,n_samp,xmin,xmax
    total_energy = []; total_nu = []

    init_photons(n_samp,xmin,xmax)
    
    for i in range(n_time_steps):
        tstop,E_save,nu_save = step(tstart)
        tstart = tstop
        print('i tstart: ',i,tstart)
        if len(E_save) > 0:
            total_energy.append(E_save); total_nu.append(nu_save)
