import numpy as np

def los_to_HI_axis_proj(dens,temp,vel,vchannel,memlim=1.,
                        deltas=1.,los_axis=1,verbose=False):
    """
        inputs:
            dens: number density of hydrogen in units of 1/cm^3
            temp: temperature in units of K
            vel: line-of-sight velocity in units of km/s
            vchannel: velocity channel in km/s
        parameters:
            deltas: length of line segments in units of pc
            memlim: memory limit in GB
            los_axis: 0 -- z, 1 -- y, 2 -- x
        outputs: a dictionary
            TB: the brightness temperature
            tau: optical depth
    """

    Nv=len(vchannel)
    mem=np.prod(temp.shape)*8*Nv/1024.**3
    if memlim>0:
        if mem>memlim:
            nchunk=int(mem/memlim)+1
            vchunk=[]
            dv=int(Nv/nchunk)+1
            if dv == 1:
                print("at least {} GB memory would be required".format(mem/Nv))
            for i in np.arange(nchunk):
                imin=i*dv
                imax=(i+1)*dv
                if imin>Nv:
                    break
                if imax>Nv:
                    vchunk.append(vchannel[imin:Nv])
                    break
                vchunk.append(vchannel[imin:imax])
        Tlos=temp[np.newaxis,...]
        vlos=vel[np.newaxis,...]
        nlos=dens[np.newaxis,...]
    else:
        Tlos=temp
        vlos=vel
        nlos=dens

    Tspin=Tlos

    ds=deltas*3.085677581467192e+18

    v_L=0.21394414*np.sqrt(Tlos) # in units of km/s
    if mem>memlim:
        TB=[]
        tau_v=[]
        if memlim>0:
            for vch in vchunk:
                if verbose: print(vch)
                phi_v=0.00019827867/v_L*np.exp(-(1.6651092223153954*
                      (vch[:,np.newaxis,np.newaxis,np.newaxis]-vlos)/v_L)**2) # time
                kappa_v=2.6137475e-15*nlos/Tspin*phi_v # area/volume = 1/length
                tau_los=kappa_v*ds # dimensionless

                tau_cumul=tau_los.cumsum(axis=los_axis+1)

                TB.append(np.nansum(Tspin*(1-np.exp(-tau_los))*np.exp(-tau_cumul),axis=los_axis+1)) # same unit with Tspin
                tau_v.append(np.nansum(kappa_v*ds,axis=los_axis+1)) # dimensionless
            TB=np.concatenate(TB,axis=0)
            tau_v=np.concatenate(tau_v,axis=0)
        else:
            for vch in vchannel:
                if verbose: print(vch)
                phi_v=0.00019827867/v_L*np.exp(-(1.6651092223153954*
                      (vch-vlos)/v_L)**2) # time
                kappa_v=2.6137475e-15*nlos/Tspin*phi_v # area/volume = 1/length
                tau_los=kappa_v*ds # dimensionless

                tau_cumul=tau_los.cumsum(axis=los_axis)

                TB.append(np.nansum(Tspin*(1-np.exp(-tau_los))*np.exp(-tau_cumul),axis=los_axis)) # same unit with Tspin
                tau_v.append(np.nansum(kappa_v*ds,axis=los_axis)) # dimensionless
    else:
        phi_v=0.00019827867/v_L*np.exp(-(1.6651092223153954*
              (vchannel[:,np.newaxis,np.newaxis,np.newaxis]-vlos)/v_L)**2) # time
        kappa_v=2.6137475e-15*nlos/Tspin*phi_v # area/volume = 1/length
        tau_los=kappa_v*ds # dimensionless

        tau_cumul=tau_los.cumsum(axis=los_axis+1)

        TB=np.nansum(Tspin*(1-np.exp(-tau_los))*np.exp(-tau_cumul),axis=los_axis+1) # same unit with Tspin
        tau_v=np.nansum(kappa_v*ds,axis=los_axis+1) # dimensionless

    return np.array(TB),np.array(tau_v)


def k10h(T2):
    """
       input: T/100K
       output: collisional excitation rate of hydrogen in c.g.s.
    """
    k10_1=1.19e-10*T2**(0.74-0.20*np.log(T2))
    k10_2=2.24e-10*T2**0.207*np.exp(-0.876/T2)
    k10=k10_1
    idx=k10_2 > k10_1
    k10[idx]=k10_2[idx]
    return k10

def k10e(T2):
    """
       input: T/100K
       output: collisional excitation rate of electrion in c.g.s.
    """
    Temp=T2*1.e2
    k10=-9.607+0.5*np.log10(Temp)*np.exp(-(np.log10(Temp))**4.5/1.8e3)
    return 10.**k10

# save synthetic HI data to fits
from astropy.io import fits

def create_fits(domain):
    hdr = fits.Header()
    hdr['time']=domain['time']
    hdr['xmin']=(domain['left_edge'][0],'pc')
    hdr['xmax']=(domain['right_edge'][0],'pc')
    hdr['ymin']=(domain['left_edge'][1],'pc')
    hdr['ymax']=(domain['right_edge'][1],'pc')
    hdr['zmin']=(domain['left_edge'][2],'pc')
    hdr['zmax']=(domain['right_edge'][2],'pc')
    hdr['dx']=(domain['dx'][0],'pc')
    hdr['dy']=(domain['dx'][1],'pc')
    hdr['dz']=(domain['dx'][2],'pc')
    hdu = fits.PrimaryHDU(header=hdr)
    return hdu

def add_header_for_glue(hdu,hdr,axis='xyz'):
    for i,ax in enumerate(axis):
        hdu.header['CDELT{}'.format(i+1)]=hdr['d{}'.format(ax)]
        hdu.header['CTYPE{}'.format(i+1)]=ax
        hdu.header['CUNIT{}'.format(i+1)]=hdr.comments['d{}'.format(ax)]
        hdu.header['CRVAL{}'.format(i+1)]=hdr['{}min'.format(ax)]
        hdu.header['CRPIX{}'.format(i+1)]=hdr['{}max'.format(ax)]+hdr['{}min'.format(ax)]
    return

def save_to_fits(domain,vchannel,TB,tau,fitsname=None):
    hdul = fits.HDUList()
    hdu = create_fits(domain)

    hdu.header['vmin']=(vchannel.min(),'km/s')
    hdu.header['vmax']=(vchannel.max(),'km/s')
    hdu.header['dv']=(vchannel[1]-vchannel[0],'km/s')

    hdul.append(hdu)
    for fdata,label in zip([TB,tau],['TB','tau']):
        hdul.append(fits.ImageHDU(name=label,data=fdata))

    hdr=hdu.header
    for hdu in hdul:
        add_header_for_glue(hdu,hdr,axis='xyv')

    if fitsname is not None: hdul.writeto(fitsname,overwrite=True)
    return hdul
