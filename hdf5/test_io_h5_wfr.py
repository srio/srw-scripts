from srwlib import *

import numpy
from numpy.testing import assert_array_almost_equal

import h5py

from srwlibAux import save_wfr_2_hdf5,_SRW_2_Numpy

def get_srw_wfr():
    #############################################################################
    # SRWLIB Example#7: Simulating propagation of a Gaussian X-ray beam through a simple optical scheme containing CRL
    # v 0.07
    #############################################################################
    print('SRWLIB Python Example # 7:')
    print('Simulating propagation of a Gaussian X-ray beam')

    #**********************Input Parameters and structures:

    # strDataFolderName = 'data_example_07' #data sub-folder name
    # strIntOutFileName1 = 'ex07_res_int_in.dat' #file name for output SR intensity data
    # strPhOutFileName1 = 'ex07_res_ph_in.dat' #file name for output SR phase data
    # strIntPropOutFileName1 = 'ex07_res_int_dist_crl_p1.dat' #file name for output SR intensity data
    # strPhPropOutFileName1 = 'ex07_res_ph_dist_crl_p1.dat' #file name for output SR phase data
    # strIntPropOutFileName2 = 'ex07_res_int_dist_crl_p2.dat' #file name for output SR intensity data
    # strPhPropOutFileName2 = 'ex07_res_ph_dist_crl_p2.dat' #file name for output SR phase data
    # strIntPropOutFileName3 = 'ex07_res_int_perf_crl_p2.dat' #file name for output SR intensity data
    # strPhPropOutFileName3 = 'ex07_res_ph_perf_crl_p2.dat' #file name for output SR phase data
    # strOpPathOutFileName1 = 'ex07_res_opt_path_dif_dist_crl.dat' #file name for output SR intensity data
    # strOpPathOutFileName2 = 'ex07_res_opt_path_dif_perf_crl.dat' #file name for output SR intensity data

    firstHorAp  = 4.e-03    #First Aperture [m]
    firstVertAp = 3.e-03    #[m]
    avgPhotEn = 12400 #5000 #Photon Energy [eV]
    sigX = 50e-06/2.35      #Horiz. RMS size at Waist [m]
    sigY = 23e-06/2.35     #Vert. RMS size at Waist [m]
    sampFactNxNyForProp = 5 #sampling factor for adjusting nx, ny (effective if > 0)
    zStart = 300 #Longitudinal Position [m] at which Electric Field has to be calculated, i.e. the position of the first optical element

    GsnBm = SRWLGsnBm() #Gaussian Beam structure (just parameters)
    GsnBm.x = 0 #Transverse Coordinates of Gaussian Beam Center at Waist [m]
    GsnBm.y = 0
    GsnBm.z = 0 #Longitudinal Coordinate of Waist [m]
    GsnBm.xp = 0 #Average Angles of Gaussian Beam at Waist [rad]
    GsnBm.yp = 0
    GsnBm.avgPhotEn = avgPhotEn #5000 #Photon Energy [eV]
    GsnBm.pulseEn = 0.001 #Energy per Pulse [J] - to be corrected
    GsnBm.repRate = 1 #Rep. Rate [Hz] - to be corrected
    GsnBm.polar = 1 #1- linear hoirizontal
    GsnBm.sigX = sigX  #Horiz. RMS size at Waist [m]
    GsnBm.sigY = sigY #Vert. RMS size at Waist [m]
    GsnBm.sigT = 10e-15 #Pulse duration [s] (not used?)
    GsnBm.mx = 2 #Transverse Gauss-Hermite Mode Orders
    GsnBm.my = 0

    wfr = SRWLWfr() #Initial Electric Field Wavefront
    wfr.allocate(1, 100, 100) #Numbers of points vs Photon Energy (1), Horizontal and Vertical Positions (dummy)

    wfr.mesh.zStart = zStart  #Longitudinal Position [m] at which Electric Field has to be calculated, i.e. the position of the first optical element
    wfr.mesh.eStart = GsnBm.avgPhotEn #Initial Photon Energy [eV]
    wfr.mesh.eFin   = GsnBm.avgPhotEn #Final Photon Energy [eV]
    wfr.mesh.xStart = -0.5*firstHorAp #Initial Horizontal Position [m]
    wfr.mesh.xFin   =  0.5*firstHorAp #Final Horizontal Position [m]
    wfr.mesh.yStart = -0.5*firstVertAp #Initial Vertical Position [m]
    wfr.mesh.yFin   =  0.5*firstVertAp #Final Vertical Position [m]

    wfr.partBeam.partStatMom1.x  = GsnBm.x #Some information about the source in the Wavefront structure
    wfr.partBeam.partStatMom1.y  = GsnBm.y
    wfr.partBeam.partStatMom1.z  = GsnBm.z
    wfr.partBeam.partStatMom1.xp = GsnBm.xp
    wfr.partBeam.partStatMom1.yp = GsnBm.yp


    arPrecPar = [sampFactNxNyForProp]

    #**********************Calculating Initial Wavefront
    srwl.CalcElecFieldGaussian(wfr, GsnBm, arPrecPar)

    return wfr

    print('done creating SRW wavefront')

def get_from_h5file(h5filename,h5filepath):
    f = h5py.File(h5filename, 'r')
    try:
        out = f[h5filepath].value
    except:
        print("Error accessing data path: %s in file %s"%(h5filepath,h5filename))
        out = None
    f.close()
    return out


def test_same_amplitudes(_wfr,h5file,h5path):

    myh5path = h5path+'/wfr_complex_amplitude_sigma'
    print("Accessing file, path",h5file,myh5path)
    amplitude_sigma = get_from_h5file(h5file,myh5path)
    print(">>>>",amplitude_sigma.shape)

    myh5path = h5path+'/wfr_complex_amplitude_pi'
    print("Accessing file, path",h5file,myh5path)
    amplitude_pi = get_from_h5file(h5file,myh5path)
    print(">>>>",amplitude_pi.shape)

    x_polarization = _SRW_2_Numpy(_wfr.arEx, _wfr.mesh.nx, _wfr.mesh.ny, _wfr.mesh.ne)   # sigma
    y_polarization = _SRW_2_Numpy(_wfr.arEy, _wfr.mesh.nx, _wfr.mesh.ny, _wfr.mesh.ne)   # pi

    e_field = numpy.concatenate((x_polarization, y_polarization), 3)

    print(">>>> sigma shapes: ", amplitude_sigma.shape,e_field[0,:,:,0].shape)
    print(">>>> pi shapes: ", amplitude_pi.shape,e_field[0,:,:,1].shape)

    assert_array_almost_equal(amplitude_sigma,e_field[0,:,:,0])
    assert_array_almost_equal(amplitude_pi,e_field[0,:,:,1])


if __name__ == "__main__":

    wfr_srw = get_srw_wfr()

    save_wfr_2_hdf5(wfr_srw,"tmp.h5",_subgroupname="wfr", _intensity=True,_amplitude=True,_phase=False,_overwrite=True)
    save_wfr_2_hdf5(wfr_srw,"tmp.h5",_subgroupname="wfr2",_intensity=False,_amplitude=True,_phase=True,_overwrite=False)

    test_same_amplitudes(wfr_srw,"tmp.h5","wfr")

