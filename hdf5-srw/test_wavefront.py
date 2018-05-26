#!/usr/bin/python
# -*- coding: utf-8 -*-
#############################################################################
# Wave propagation through ID16A - U18.3 - 17keV mode
# Author: Rafael Celestre
# Rafael.Celestre@esrf.fr
# 28.08.2017
#############################################################################


from srwlib import *
# from srw_aux_tools import tic,toc,plot_wfr,dump_open,dump_wfr,dump_close, srwl_wfr_emit_prop_multi_e_NEW


import scipy.constants as codata
import numpy
try:
    import matplotlib.pylab as plt
    plt.switch_backend("Qt5Agg")
except:
    raise Exception("Failed to set matplotlib backend to Qt5Agg")


from srw_aux_tools import plot_wfr


#
# main code
#

def calculate_source(Source="EBS",pMltLr=28.3,do_plots=True,dump_file=False):
    #############################################################################
    # Photon source
    #********************************Undulator parameters (U20.2)
    numPer = 77			# Number of ID Periods
    undPer = 0.0183		# Period Length [m]
    phB = 0	        	# Initial Phase of the Horizontal field component
    sB = 1		        # Symmetry of the Horizontal field component vs Longitudinal position
    xcID = 0 			# Transverse Coordinates of Undulator Center [m]
    ycID = 0
    zcID = 0
    n = 1
    beamE = 17
    #********************************Storage ring parameters

    # Wavelength = 1E-10*12.39841975/beamE
    Wavelength = codata.h*codata.c/codata.e/(1e3*beamE)


    # these first order moments CONTAIN the initial condition of the electron (X,X',Y,Y') (energy comes later)
    eBeam = SRWLPartBeam()
    eBeam.Iavg = 0.2             # average Current [A]
    eBeam.partStatMom1.x = 0.
    eBeam.partStatMom1.y = 0.
    eBeam.partStatMom1.z = -0.5*undPer*(numPer + 4) # initial Longitudinal Coordinate (set before the ID)
    eBeam.partStatMom1.xp = 0.   					# initial Relative Transverse Velocities
    eBeam.partStatMom1.yp = 0.

    electron_rest_energy_in_GeV = codata.electron_mass*codata.c**2/codata.e*1e-9
    KtoBfactor = codata.e/(2*pi*codata.electron_mass*codata.c)

    #
    # obviously these emittances value (with exception of the electron_energy) are not used for
    # the single electron calculation
    #
    if (Source.lower() == 'ebs'):
        # e- beam paramters (RMS) EBS
        sigEperE = 9.3E-4 			# relative RMS energy spread
        sigX  = 30.3E-06			# horizontal RMS size of e-beam [m]
        sigXp = 4.4E-06				# horizontal RMS angular divergence [rad]
        sigY  = 3.6E-06				# vertical RMS size of e-beam [m]
        sigYp = 1.46E-06			# vertical RMS angular divergence [rad]
        electron_energy_in_GeV = 6.00
        # eBeam.partStatMom1.gamma = 6.00/electron_rest_energy_in_GeV # Relative Energy
        # K = sqrt(2)*sqrt(((Wavelength*2*n*eBeam.partStatMom1.gamma**2)/undPer)-1)
        # # B = K/(undPer*93.3728962)	# Peak Horizontal field [T] (undulator)
        # B = K/(undPer*KtoBfactor)	# Peak Horizontal field [T] (undulator)

    else:
        # e- beam paramters (RMS) ESRF @ low beta
        sigEperE = 1.1E-3 			# relative RMS energy spread
        sigX     = 48.6E-06			# horizontal RMS size of e-beam [m]
        sigXp    = 106.9E-06			# horizontal RMS angular divergence [rad]
        sigY     = 3.5E-06				# vertical RMS size of e-beam [m]
        sigYp    = 1.26E-06			# vertical RMS angular divergence [rad]
        electron_energy_in_GeV = 6.04

    eBeam.partStatMom1.gamma = electron_energy_in_GeV/electron_rest_energy_in_GeV # Relative Energy
    K = sqrt(2)*sqrt(((Wavelength*2*n*eBeam.partStatMom1.gamma**2)/undPer)-1)
    # B = K/(undPer*93.3728962)	# Peak Horizontal field [T] (undulator)
    B = K/(undPer*KtoBfactor)	# Peak Horizontal field [T] (undulator)


    # 2nd order stat. moments
    eBeam.arStatMom2[0] = sigX*sigX			 # <(x-<x>)^2>
    eBeam.arStatMom2[1] = 0					 # <(x-<x>)(x'-<x'>)>
    eBeam.arStatMom2[2] = sigXp*sigXp		 # <(x'-<x'>)^2>
    eBeam.arStatMom2[3] = sigY*sigY		     # <(y-<y>)^2>
    eBeam.arStatMom2[4] = 0					 # <(y-<y>)(y'-<y'>)>
    eBeam.arStatMom2[5] = sigYp*sigYp		 # <(y'-<y'>)^2>
    eBeam.arStatMom2[10] = sigEperE*sigEperE # <(E-<E>)^2>/<E>^2

    # Electron trajectory
    eTraj = 0

    # Precision parameters
    arPrecSR = [0]*7
    arPrecSR[0] = 1		# SR calculation method: 0- "manual", 1- "auto-undulator", 2- "auto-wiggler"
    arPrecSR[1] = 0.01	# relative precision
    arPrecSR[2] = 0		# longitudinal position to start integration (effective if < zEndInteg)
    arPrecSR[3] = 0		# longitudinal position to finish integration (effective if > zStartInteg)
    arPrecSR[4] = 20000	# Number of points for trajectory calculation
    arPrecSR[5] = 1		# Use "terminating terms"  or not (1 or 0 respectively)
    arPrecSR[6] = 0		# sampling factor for adjusting nx, ny (effective if > 0)	# -1 @Petra

    sampFactNxNyForProp = 0 # sampling factor for adjusting nx, ny (effective if > 0)

    und = SRWLMagFldU([SRWLMagFldH(n, 'v', B, phB, sB, 1)], undPer, numPer)

    magFldCnt = SRWLMagFldC([und], array('d', [xcID]), array('d', [ycID]), array('d', [zcID]))

    #********************************Wavefronts
    # Monochromatic wavefront
    wfr = SRWLWfr()
    wfr.allocate(1, 512, 256)  # Photon Energy, Horizontal and Vertical Positions
    wfr.mesh.zStart = pMltLr
    wfr.mesh.eStart = beamE*1E3
    wfr.mesh.eFin = wfr.mesh.eStart
    wfr.mesh.xStart = -2.5*1E-3
    wfr.mesh.xFin = - wfr.mesh.xStart
    wfr.mesh.yStart = -1*1E-3
    wfr.mesh.yFin = - wfr.mesh.yStart
    wfr.partBeam = eBeam

    print('- Performing Initial Electric Field calculation ... ')
    srwl.CalcElecFieldSR(wfr, eTraj, magFldCnt, arPrecSR)

    #
    # plot source
    #
    if do_plots:
        arI1,x,y = plot_wfr(wfr,kind='intensity',title='Source Intensity at ' + str(wfr.mesh.eStart) + ' eV',
                 xtitle='Horizontal Position [um]',
                 ytitle='Vertical Position [um]',aspect=None,show=True)


        # arP1,x,y = plot_wfr(wfr,kind='phase',title='Source Phase at ' + str(wfr.mesh.eStart) + ' eV',
        # 		 xtitle='Horizontal Position [um]',
        # 		 ytitle='Vertical Position [um]',show=True)

    if dump_file:
        filename = "tmp.h5"
        f = dump_open(filename)
        dump_wfr(f,wfr,prefix="source")


    print('\nsource calculation finished\n')
    return wfr

def propagate_beamline(wfr,do_plots=True,dump_file=False):

    print('\nWave propagation through ID16A - U18.3 - 17keV mode\n')

    #############################################################################
    # Program variables
    defocus  = 0		  # (-) before focus, (+) after focus
    MultiE   = 0		  # Multi -e calculation: 0=Single electron, 1=SRW routine, 2=
    # dump_file    = False  # saving data
    # do_plots     = True # Generates graphical results only when MultiE is set to 'off'
    Errors   = "off"	# loads (or not - off) errors for the optics
    ThnElmnt = "on" 	# ThnElmnt = "on" uses Thin Lenses for all optical elements
    Source	 = "EBS"
    nMacroElec = 10000 # 5000	# total number of macro-electrons
    directory  = '/ID16A'
    #############################################################################
    # Files and folders names
    if (Errors.lower() == 'on'):
        oeErr = "_Err"
    else:
        oeErr = "_noErr"
    strDataFolderName = 'simulations'+directory

    if (ThnElmnt.lower() == 'on'):
        Prfx = "_TE"	#Thin Element
    else:
        Prfx = "_OE"	#Optical Element

    strDataFolderName = 'simulations'+directory
    strIntPropOutFileName  = Source+Prfx+"_d"+str(defocus)+'_AP_intensity'+oeErr+'.dat'
    strPhPropOutFileName   = Source+Prfx+"_d"+str(defocus)+'_AP_phase'+oeErr+'.dat'
    strIntPrtlChrnc 	   = Source+Prfx+"_"+str(nMacroElec/1000)+"k_d"+str(defocus)+'_ME_AP_intensity'+oeErr+'.dat'


    #############################################################################
    # Beamline assembly
    print("\nSetting up beamline\n")
    beamE = 17
    # Wavelength = 1E-10*12.39841975/beamE
    Wavelength = codata.h*codata.c/codata.e/(1e3*beamE)
    #============= ABSOLUTE POSITIONS =====================================#
    pMltLr = 28.3
    pSlt   = 40
    pKBV   = 184.90
    pKBH   = 184.95
    pGBE   = 185

    Drft1  = SRWLOptD((pSlt-pMltLr))
    Drft2  = SRWLOptD(pKBV-pSlt)
    Drft3  = SRWLOptD(pKBH-pKBV)
    Drft4  = SRWLOptD(pGBE-pKBH+defocus)

    #============= MULTILAYER(H) ==========================================#
    """Side bounce multi layer horizontal focusing mirror generating a
    secondary source on a slit upstream the beamline. Here, the mirror is
    represented as a ideal thin lens + an aperture to limit the beam
    footprint"""
    W_MltLr = 13*1E-3
    L_MltLr = 120*1E-3
    grzAngl = 31.42*1E-3
    oeAptrMltLr = SRWLOptA('r','a',L_MltLr*sin(grzAngl),W_MltLr)

    if (ThnElmnt.lower() == 'on'):
        fMltLrh = 1/((1/pMltLr)+(1/(pSlt-pMltLr)))
        fMltLrv =1E23
        oeMltLr = SRWLOptL(_Fx=fMltLrh, _Fy=fMltLrv)

    else:
        oeMltLr = SRWLOptMirEl(_p = pMltLr,_q=(pSlt-pMltLr) ,_ang_graz=grzAngl,_r_sag=1E23,	_size_tang=L_MltLr, _size_sag=W_MltLr, _nvx=cos(grzAngl), _nvy=0, _nvz=-sin(grzAngl), _tvx=-sin(grzAngl), _tvy=0)
        #if (Errors.lower() == 'on'):
            # Loading 1D mirror slope error from .dat file
            #heightProfData = srwl_uti_read_data_cols(os.path.join(os.getcwd(), strDataFolderName, strErrFldrName, strMirSrfHghtErrorML), _str_sep='\t', _i_col_start=0, _i_col_end=1)
            #oeM1err = srwl_opt_setup_surf_height_1d(heightProfData, _dim='x', _ang=GrzngAngl, _amp_coef=1) # _dim='x' for side bounce
    #============= VIRTUAL SOURCE SLIT ====================================#
    VSS_h = 50*1E-6
    VSS_v = 5000*1E-6
    oeVSS = SRWLOptA('r','a',VSS_h,VSS_v)
    #============= KB(V) ==================================================#
    """The fisrt mirror of the KB system has a vertical focusing function.
    It takes the beam from the source and focuses if onto the sample. In our
    convention, bounce down for for vertical focusing elements."""
    W_KBv = 50*1E-3
    L_KBv = 60*1E-3
    grzAngl = 14.99*1E-3
    oeAptrKBv = SRWLOptA('r','a',W_KBv,L_KBv*sin(grzAngl))

    if (ThnElmnt.lower() == 'on'):
        fKBv_v = 1/((1/pKBV)+(1/(pGBE-pKBV)))
        fKBv_h =1E23
        oeKBv = SRWLOptL(_Fx=fKBv_h, _Fy=fKBv_v)

    else:
        oeKBv = SRWLOptMirEl(_p = pKBV,_q=(pGBE-pKBV) ,_ang_graz=grzAngl,_r_sag=1E23,_size_tang=L_KBv, _size_sag=W_KBv, _nvx=0, _nvy=cos(grzAngl), _nvz=-sin(grzAngl), _tvx=0, _tvy=-sin(grzAngl))
        #if (Errors.lower() == 'on'):
            # Loading 1D mirror slope error from .dat file
            #heightProfData = srwl_uti_read_data_cols(os.path.join(os.getcwd(), strDataFolderName, strErrFldrName, strMirSrfHghtError), _str_sep='\t', _i_col_start=0, _i_col_end=1)
            #oeM1err = srwl_opt_setup_surf_height_1d(heightProfData, _dim='x', _ang=GrzngAngl, _amp_coef=1)
    #============= KB(H) ==================================================#
    """The second mirror of the KB system has a horisontal focusing function.
    It takes the beam from the virtual source (VSS) and focuses if onto the
    sample. In our convention, side bounce for for horizontal focusing
    elements."""
    W_KBh = 13*1E-3
    L_KBh  = 26*1E-3
    grzAngl = 14.99*1E-3
    oeAptrKBh = SRWLOptA('r','a',L_KBh*sin(grzAngl),W_KBh)

    if (ThnElmnt.lower() == 'on'):
        fKBh_v = 1E23
        fKBh_h = 1/((1/(pKBH-pSlt))+(1/(pGBE-pKBH)))
        oeKBh = SRWLOptL(_Fx=fKBh_h, _Fy=fKBh_v)

    else:
        oeKBh = SRWLOptMirEl(_p=(pKBH-pSlt),_q=(pGBE-pKBH),_ang_graz=grzAngl,_r_sag=1E23,_size_tang=L_KBh, _size_sag=W_KBh, _nvx=cos(grzAngl), _nvy=0, _nvz=-sin(grzAngl), _tvx=-sin(grzAngl), _tvy=0)
        #if (Errors.lower() == 'on'):
            # Loading 1D mirror slope error from .dat file
            #heightProfData = srwl_uti_read_data_cols(os.path.join(os.getcwd(), strDataFolderName, strErrFldrName, strMirSrfHghtError), _str_sep='\t', _i_col_start=0, _i_col_end=1)
            #oeM1err = srwl_opt_setup_surf_height_1d(heightProfData, _dim='x', _ang=GrzngAngl, _amp_coef=1)	# _dim='x' for side bounce


    #============= Wavefront Propagation Parameters =======================#
    #                [ 0] [1] [2]  [3]  [4]  [5]  [6]  [7]  [8]  [9] [10] [11]
    ppAptrMltLr		=[ 0,  0, 1.,   0,   0,  1.,  4.,  1.,  4.,   0,   0,   0]
    ppMltLr			=[ 0,  0, 1.,   0,   0,  1.,  1.,  1.,  1.,   0,   0,   0]
    ppDrft1 		=[ 0,  0, 1.,   2,   0,  1.,  1.,  1.,  1.,   0,   0,   0]
    ppVSS			=[ 0,  0, 1.,   0,   0,0.70,1.43,  1.,  1.,   0,   0,   0]
    ppDrft2			=[ 0,  0, 1.,   2,   0,  1.,  1,   1.,  1.,   0,   0,   0]
    ppAptrKBv		=[ 0,  0, 1.,   0,   0,  1.,  1.,  1.,  1.,   0,   0,   0]
    ppKBv			=[ 0,  0, 1.,   0,   0, 0.5,  2.,0.08,12.5,   0,   0,   0]
    ppDrft3			=[ 0,  0, 1.,   1,   0,  1.,  1.,  1.,  1.,   0,   0,   0]
    ppAptrKBh		=[ 0,  0, 1.,   0,   0,  1.,  1.,  1.,  1.,   0,   0,   0]
    ppKBh			=[ 0,  0, 1.,   0,   0,0.025,40.,  1.,  1.,   0,   0,   0]
    ppDrft4			=[ 0,  0, 1.,   1,   0,  1.,  1.,  1.,  1.,   0,   0,   0]
    ppFinal			=[ 0,  0, 1.,   0,   1,0.14,  8.,0.10, 10.,   0,   0,   0]

    optBL = SRWLOptC(
                    [oeAptrMltLr,   oeMltLr,   Drft1,    ], # oeVSS,     Drft2, oeAptrKBv,oeKBv,  Drft3, oeAptrKBh, oeKBh,  Drft4],
                    [ppAptrMltLr,   ppMltLr, ppDrft1,    ], # ppVSS,   ppDrft2, ppAptrKBv,ppKBv,ppDrft3, ppAptrKBh, ppKBh,ppDrft4,ppFinal],
                     )

	# """
	# [ 0]: Auto-Resize (1) or not (0) Before propagation
	# [ 1]: Auto-Resize (1) or not (0) After propagation
	# [ 2]: Relative Precision for propagation with Auto-Resizing (1. is nominal)
	# [ 3]: Type of Free-Space Propagator:
	# 	   0- Standard Fresnel
	# 	   1- Fresnel with analytical treatment of the quadratic (leading) phase terms
	# 	   2- Similar to 1, yet with different processing near a waist
	# 	   3- For propagation from a waist over a ~large distance
	# 	   4- For propagation over some distance to a waist
	# [ 4]: Do any Resizing on Fourier side, using FFT, (1) or not (0)
	# [ 5]: Horizontal Range modification factor at Resizing (1. means no modification)
	# [ 6]: Horizontal Resolution modification factor at Resizing
	# [ 7]: Vertical Range modification factor at Resizing
	# [ 8]: Vertical Resolution modification factor at Resizing
	# [ 9]: Type of wavefront Shift before Resizing (not yet implemented)
	# [10]: New Horizontal wavefront Center position after Shift (not yet implemented)
	# [11]: New Vertical wavefront Center position after Shift (not yet implemented)
	# """

    srwl.PropagElecField(wfr, optBL)

    if do_plots:
        arI1,x,y = plot_wfr(wfr,kind='intensity',title='Focal Intensity at ' + str(wfr.mesh.eStart) + ' eV',
                 xtitle='Horizontal Position [um]',
                 ytitle='Vertical Position [um]',show=True)


        # arP1,x,y = plot_wfr(wfr,kind='phase',title='Focal Phase at ' + str(wfr.mesh.eStart) + ' eV',
        # 		 xtitle='Horizontal Position [um]',
        # 		 ytitle='Vertical Position [um]',show=True)

    if dump_file:
        dump_wfr(f,wfr,prefix="focus")
        dump_close(f)
        print("File written to disk: %s"%(filename))

    print('\ncalculation finished\n')
    return wfr


# def main():
#
# 	tic()
#
# 	print('\nWave propagation through ID16A - U18.3 - 17keV mode\n')
#
# 	#############################################################################
# 	# Program variables
# 	defocus  = 0		  # (-) before focus, (+) after focus
# 	MultiE   = 0		  # Multi -e calculation: 0=Single electron, 1=SRW routine, 2=
# 	dump_file    = False  # saving data
# 	do_plots     = True # Generates graphical results only when MultiE is set to 'off'
# 	Errors   = "off"	# loads (or not - off) errors for the optics
# 	ThnElmnt = "on" 	# ThnElmnt = "on" uses Thin Lenses for all optical elements
# 	Source	 = "EBS"
# 	nMacroElec = 10000 # 5000	# total number of macro-electrons
# 	directory  = '/ID16A'
# 	#############################################################################
# 	# Files and folders names
# 	if (Errors.lower() == 'on'):
# 		oeErr = "_Err"
# 	else:
# 		oeErr = "_noErr"
# 	strDataFolderName = 'simulations'+directory
#
# 	if (ThnElmnt.lower() == 'on'):
# 		Prfx = "_TE"	#Thin Element
# 	else:
# 		Prfx = "_OE"	#Optical Element
#
# 	strDataFolderName = 'simulations'+directory
# 	strIntPropOutFileName  = Source+Prfx+"_d"+str(defocus)+'_AP_intensity'+oeErr+'.dat'
# 	strPhPropOutFileName   = Source+Prfx+"_d"+str(defocus)+'_AP_phase'+oeErr+'.dat'
# 	strIntPrtlChrnc 	   = Source+Prfx+"_"+str(nMacroElec/1000)+"k_d"+str(defocus)+'_ME_AP_intensity'+oeErr+'.dat'
#
#
# 	print(Source + ' lattice, ' + Prfx +' optics, ' + oeErr + ' optics errors\n')
#
# 	#############################################################################
# 	# Beamline assembly
# 	print("\nSetting up beamline\n")
# 	beamE = 17
# 	# Wavelength = 1E-10*12.39841975/beamE
# 	Wavelength = codata.h*codata.c/codata.e/(1e3*beamE)
# 	#============= ABSOLUTE POSITIONS =====================================#
# 	pMltLr = 28.3
# 	pSlt   = 40
# 	pKBV   = 184.90
# 	pKBH   = 184.95
# 	pGBE   = 185
#
# 	Drft1  = SRWLOptD((pSlt-pMltLr))
# 	Drft2  = SRWLOptD(pKBV-pSlt)
# 	Drft3  = SRWLOptD(pKBH-pKBV)
# 	Drft4  = SRWLOptD(pGBE-pKBH+defocus)
#
# 	#============= MULTILAYER(H) ==========================================#
# 	"""Side bounce multi layer horizontal focusing mirror generating a
# 	secondary source on a slit upstream the beamline. Here, the mirror is
# 	represented as a ideal thin lens + an aperture to limit the beam
# 	footprint"""
# 	W_MltLr = 13*1E-3
# 	L_MltLr = 120*1E-3
# 	grzAngl = 31.42*1E-3
# 	oeAptrMltLr = SRWLOptA('r','a',L_MltLr*sin(grzAngl),W_MltLr)
#
# 	if (ThnElmnt.lower() == 'on'):
# 		fMltLrh = 1/((1/pMltLr)+(1/(pSlt-pMltLr)))
# 		fMltLrv =1E23
# 		oeMltLr = SRWLOptL(_Fx=fMltLrh, _Fy=fMltLrv)
#
# 	else:
# 		oeMltLr = SRWLOptMirEl(_p = pMltLr,_q=(pSlt-pMltLr) ,_ang_graz=grzAngl,_r_sag=1E23,	_size_tang=L_MltLr, _size_sag=W_MltLr, _nvx=cos(grzAngl), _nvy=0, _nvz=-sin(grzAngl), _tvx=-sin(grzAngl), _tvy=0)
# 		#if (Errors.lower() == 'on'):
# 			# Loading 1D mirror slope error from .dat file
# 			#heightProfData = srwl_uti_read_data_cols(os.path.join(os.getcwd(), strDataFolderName, strErrFldrName, strMirSrfHghtErrorML), _str_sep='\t', _i_col_start=0, _i_col_end=1)
# 			#oeM1err = srwl_opt_setup_surf_height_1d(heightProfData, _dim='x', _ang=GrzngAngl, _amp_coef=1) # _dim='x' for side bounce
# 	#============= VIRTUAL SOURCE SLIT ====================================#
# 	VSS_h = 50*1E-6
# 	VSS_v = 5000*1E-6
# 	oeVSS = SRWLOptA('r','a',VSS_h,VSS_v)
# 	#============= KB(V) ==================================================#
# 	"""The fisrt mirror of the KB system has a vertical focusing function.
# 	It takes the beam from the source and focuses if onto the sample. In our
# 	convention, bounce down for for vertical focusing elements."""
# 	W_KBv = 50*1E-3
# 	L_KBv = 60*1E-3
# 	grzAngl = 14.99*1E-3
# 	oeAptrKBv = SRWLOptA('r','a',W_KBv,L_KBv*sin(grzAngl))
#
# 	if (ThnElmnt.lower() == 'on'):
# 		fKBv_v = 1/((1/pKBV)+(1/(pGBE-pKBV)))
# 		fKBv_h =1E23
# 		oeKBv = SRWLOptL(_Fx=fKBv_h, _Fy=fKBv_v)
#
# 	else:
# 		oeKBv = SRWLOptMirEl(_p = pKBV,_q=(pGBE-pKBV) ,_ang_graz=grzAngl,_r_sag=1E23,_size_tang=L_KBv, _size_sag=W_KBv, _nvx=0, _nvy=cos(grzAngl), _nvz=-sin(grzAngl), _tvx=0, _tvy=-sin(grzAngl))
# 		#if (Errors.lower() == 'on'):
# 			# Loading 1D mirror slope error from .dat file
# 			#heightProfData = srwl_uti_read_data_cols(os.path.join(os.getcwd(), strDataFolderName, strErrFldrName, strMirSrfHghtError), _str_sep='\t', _i_col_start=0, _i_col_end=1)
# 			#oeM1err = srwl_opt_setup_surf_height_1d(heightProfData, _dim='x', _ang=GrzngAngl, _amp_coef=1)
# 	#============= KB(H) ==================================================#
# 	"""The second mirror of the KB system has a horisontal focusing function.
# 	It takes the beam from the virtual source (VSS) and focuses if onto the
# 	sample. In our convention, side bounce for for horizontal focusing
# 	elements."""
# 	W_KBh = 13*1E-3
# 	L_KBh  = 26*1E-3
# 	grzAngl = 14.99*1E-3
# 	oeAptrKBh = SRWLOptA('r','a',L_KBh*sin(grzAngl),W_KBh)
#
# 	if (ThnElmnt.lower() == 'on'):
# 		fKBh_v = 1E23
# 		fKBh_h = 1/((1/(pKBH-pSlt))+(1/(pGBE-pKBH)))
# 		oeKBh = SRWLOptL(_Fx=fKBh_h, _Fy=fKBh_v)
#
# 	else:
# 		oeKBh = SRWLOptMirEl(_p=(pKBH-pSlt),_q=(pGBE-pKBH),_ang_graz=grzAngl,_r_sag=1E23,_size_tang=L_KBh, _size_sag=W_KBh, _nvx=cos(grzAngl), _nvy=0, _nvz=-sin(grzAngl), _tvx=-sin(grzAngl), _tvy=0)
# 		#if (Errors.lower() == 'on'):
# 			# Loading 1D mirror slope error from .dat file
# 			#heightProfData = srwl_uti_read_data_cols(os.path.join(os.getcwd(), strDataFolderName, strErrFldrName, strMirSrfHghtError), _str_sep='\t', _i_col_start=0, _i_col_end=1)
# 			#oeM1err = srwl_opt_setup_surf_height_1d(heightProfData, _dim='x', _ang=GrzngAngl, _amp_coef=1)	# _dim='x' for side bounce
#
#
# 	#============= Wavefront Propagation Parameters =======================#
# 	#                [ 0] [1] [2]  [3]  [4]  [5]  [6]  [7]  [8]  [9] [10] [11]
# 	ppAptrMltLr		=[ 0,  0, 1.,   0,   0,  1.,  4.,  1.,  4.,   0,   0,   0]
# 	ppMltLr			=[ 0,  0, 1.,   0,   0,  1.,  1.,  1.,  1.,   0,   0,   0]
# 	ppDrft1 		=[ 0,  0, 1.,   2,   0,  1.,  1.,  1.,  1.,   0,   0,   0]
# 	ppVSS			=[ 0,  0, 1.,   0,   0,0.70,1.43,  1.,  1.,   0,   0,   0]
# 	ppDrft2			=[ 0,  0, 1.,   2,   0,  1.,  1,   1.,  1.,   0,   0,   0]
# 	ppAptrKBv		=[ 0,  0, 1.,   0,   0,  1.,  1.,  1.,  1.,   0,   0,   0]
# 	ppKBv			=[ 0,  0, 1.,   0,   0, 0.5,  2.,0.08,12.5,   0,   0,   0]
# 	ppDrft3			=[ 0,  0, 1.,   1,   0,  1.,  1.,  1.,  1.,   0,   0,   0]
# 	ppAptrKBh		=[ 0,  0, 1.,   0,   0,  1.,  1.,  1.,  1.,   0,   0,   0]
# 	ppKBh			=[ 0,  0, 1.,   0,   0,0.025,40.,  1.,  1.,   0,   0,   0]
# 	ppDrft4			=[ 0,  0, 1.,   1,   0,  1.,  1.,  1.,  1.,   0,   0,   0]
# 	ppFinal			=[ 0,  0, 1.,   0,   1,0.14,  8.,0.10, 10.,   0,   0,   0]
#
# 	optBL = SRWLOptC(
# 					[oeAptrMltLr,   oeMltLr,   Drft1,    ], # oeVSS,     Drft2, oeAptrKBv,oeKBv,  Drft3, oeAptrKBh, oeKBh,  Drft4],
# 					[ppAptrMltLr,   ppMltLr, ppDrft1,    ], # ppVSS,   ppDrft2, ppAptrKBv,ppKBv,ppDrft3, ppAptrKBh, ppKBh,ppDrft4,ppFinal],
# 					 )
#
# 	"""
# 	[ 0]: Auto-Resize (1) or not (0) Before propagation
# 	[ 1]: Auto-Resize (1) or not (0) After propagation
# 	[ 2]: Relative Precision for propagation with Auto-Resizing (1. is nominal)
# 	[ 3]: Type of Free-Space Propagator:
# 		   0- Standard Fresnel
# 		   1- Fresnel with analytical treatment of the quadratic (leading) phase terms
# 		   2- Similar to 1, yet with different processing near a waist
# 		   3- For propagation from a waist over a ~large distance
# 		   4- For propagation over some distance to a waist
# 	[ 4]: Do any Resizing on Fourier side, using FFT, (1) or not (0)
# 	[ 5]: Horizontal Range modification factor at Resizing (1. means no modification)
# 	[ 6]: Horizontal Resolution modification factor at Resizing
# 	[ 7]: Vertical Range modification factor at Resizing
# 	[ 8]: Vertical Resolution modification factor at Resizing
# 	[ 9]: Type of wavefront Shift before Resizing (not yet implemented)
# 	[10]: New Horizontal wavefront Center position after Shift (not yet implemented)
# 	[11]: New Vertical wavefront Center position after Shift (not yet implemented)
# 	"""
# 	#############################################################################
# 	# Photon source
#
# 	#********************************Undulator parameters (U20.2)
# 	numPer = 77			# Number of ID Periods
# 	undPer = 0.0183		# Period Length [m]
# 	phB = 0	        	# Initial Phase of the Horizontal field component
# 	sB = 1		        # Symmetry of the Horizontal field component vs Longitudinal position
# 	xcID = 0 			# Transverse Coordinates of Undulator Center [m]
# 	ycID = 0
# 	zcID = 0
# 	n = 1
# 	#********************************Storage ring parameters
#
# 	# these first order moments CONTAIN the initial condition of the electron (X,X',Y,Y') (energy comes later)
# 	eBeam = SRWLPartBeam()
# 	eBeam.Iavg = 0.2             # average Current [A]
# 	eBeam.partStatMom1.x = 0.
# 	eBeam.partStatMom1.y = 0.
# 	eBeam.partStatMom1.z = -0.5*undPer*(numPer + 4) # initial Longitudinal Coordinate (set before the ID)
# 	eBeam.partStatMom1.xp = 0.   					# initial Relative Transverse Velocities
# 	eBeam.partStatMom1.yp = 0.
#
# 	electron_rest_energy_in_GeV = codata.electron_mass*codata.c**2/codata.e*1e-9
# 	KtoBfactor = codata.e/(2*pi*codata.electron_mass*codata.c)
#
# 	#
# 	# obviously these emittances value (with exception of the electron_energy) are not used for
# 	# the single electron calculation
# 	#
# 	if (Source.lower() == 'ebs'):
# 		# e- beam paramters (RMS) EBS
# 		sigEperE = 9.3E-4 			# relative RMS energy spread
# 		sigX  = 30.3E-06			# horizontal RMS size of e-beam [m]
# 		sigXp = 4.4E-06				# horizontal RMS angular divergence [rad]
# 		sigY  = 3.6E-06				# vertical RMS size of e-beam [m]
# 		sigYp = 1.46E-06			# vertical RMS angular divergence [rad]
# 		electron_energy_in_GeV = 6.00
# 		# eBeam.partStatMom1.gamma = 6.00/electron_rest_energy_in_GeV # Relative Energy
# 		# K = sqrt(2)*sqrt(((Wavelength*2*n*eBeam.partStatMom1.gamma**2)/undPer)-1)
# 		# # B = K/(undPer*93.3728962)	# Peak Horizontal field [T] (undulator)
# 		# B = K/(undPer*KtoBfactor)	# Peak Horizontal field [T] (undulator)
#
# 	else:
# 		# e- beam paramters (RMS) ESRF @ low beta
# 		sigEperE = 1.1E-3 			# relative RMS energy spread
# 		sigX     = 48.6E-06			# horizontal RMS size of e-beam [m]
# 		sigXp    = 106.9E-06			# horizontal RMS angular divergence [rad]
# 		sigY     = 3.5E-06				# vertical RMS size of e-beam [m]
# 		sigYp    = 1.26E-06			# vertical RMS angular divergence [rad]
# 		electron_energy_in_GeV = 6.04
#
# 	eBeam.partStatMom1.gamma = electron_energy_in_GeV/electron_rest_energy_in_GeV # Relative Energy
# 	K = sqrt(2)*sqrt(((Wavelength*2*n*eBeam.partStatMom1.gamma**2)/undPer)-1)
# 	# B = K/(undPer*93.3728962)	# Peak Horizontal field [T] (undulator)
# 	B = K/(undPer*KtoBfactor)	# Peak Horizontal field [T] (undulator)
#
#
# 	# 2nd order stat. moments
# 	eBeam.arStatMom2[0] = sigX*sigX			 # <(x-<x>)^2>
# 	eBeam.arStatMom2[1] = 0					 # <(x-<x>)(x'-<x'>)>
# 	eBeam.arStatMom2[2] = sigXp*sigXp		 # <(x'-<x'>)^2>
# 	eBeam.arStatMom2[3] = sigY*sigY		     # <(y-<y>)^2>
# 	eBeam.arStatMom2[4] = 0					 # <(y-<y>)(y'-<y'>)>
# 	eBeam.arStatMom2[5] = sigYp*sigYp		 # <(y'-<y'>)^2>
# 	eBeam.arStatMom2[10] = sigEperE*sigEperE # <(E-<E>)^2>/<E>^2
#
# 	# Electron trajectory
# 	eTraj = 0
#
# 	# Precision parameters
# 	arPrecSR = [0]*7
# 	arPrecSR[0] = 1		# SR calculation method: 0- "manual", 1- "auto-undulator", 2- "auto-wiggler"
# 	arPrecSR[1] = 0.01	# relative precision
# 	arPrecSR[2] = 0		# longitudinal position to start integration (effective if < zEndInteg)
# 	arPrecSR[3] = 0		# longitudinal position to finish integration (effective if > zStartInteg)
# 	arPrecSR[4] = 20000	# Number of points for trajectory calculation
# 	arPrecSR[5] = 1		# Use "terminating terms"  or not (1 or 0 respectively)
# 	arPrecSR[6] = 0		# sampling factor for adjusting nx, ny (effective if > 0)	# -1 @Petra
#
# 	sampFactNxNyForProp = 0 # sampling factor for adjusting nx, ny (effective if > 0)
#
# 	und = SRWLMagFldU([SRWLMagFldH(n, 'v', B, phB, sB, 1)], undPer, numPer)
#
# 	magFldCnt = SRWLMagFldC([und], array('d', [xcID]), array('d', [ycID]), array('d', [zcID]))
#
# 	#********************************Wavefronts
# 	# Monochromatic wavefront
# 	wfr = SRWLWfr()
# 	wfr.allocate(1, 256, 256)  # Photon Energy, Horizontal and Vertical Positions
# 	wfr.mesh.zStart = pMltLr
# 	wfr.mesh.eStart = beamE*1E3
# 	wfr.mesh.eFin = wfr.mesh.eStart
# 	wfr.mesh.xStart = -2.5*1E-3
# 	wfr.mesh.xFin = - wfr.mesh.xStart
# 	wfr.mesh.yStart = -1*1E-3
# 	wfr.mesh.yFin = - wfr.mesh.yStart
# 	wfr.partBeam = eBeam
#
# 	meshPartCoh = deepcopy(wfr.mesh)
#
# 	#############################################################################
# 	# Wavefront generation and beam propagation
#
# 	wfr0 = deepcopy(wfr)
#
# 	if(srwl_uti_proc_is_master()):
#
# 		#********************************Calculating Initial Wavefront and extracting Intensity:
#
#
# 		# i= 0 Electron Coord.: x= -5.31942961516259e-05 x'= -4.256161067209395e-06 y= -6.985445582705632e-07 y'= 8.357970620475086e-07 E= 6.002326228169875
# 		# i= 1 Electron Coord.: x= -2.7074235691140736e-05 x'= 5.168259996376612e-06 y= -6.825593892060933e-07 y'= -2.2658468477885813e-07 E= 6.004500995161671
# 		# i= 2 Electron Coord.: x= -4.648607227005506e-06 x'= -1.8516748893362065e-06 y= 1.208825514277266e-07 y'= -5.517433654137355e-08 E= 5.992354787190328
# 		# X =   [-5.31942961516259e-05,-2.7074235691140736e-05,-4.648607227005506e-06]
# 		# Y =   [-6.985445582705632e-07,-6.825593892060933e-07,1.208825514277266e-07]
# 		# XP =  [-4.256161067209395e-06,5.168259996376612e-06 ,-1.8516748893362065e-06]
# 		# YP =  [8.357970620475086e-07,-2.2658468477885813e-07 ,-5.517433654137355e-08]
# 		# E =   [ 6.002326228169875,6.004500995161671,5.992354787190328]
#         #
# 		# wfr.partBeam.partStatMom1.x     =  X[0]  # numpy.random.normal(0.0,sigX)
# 		# wfr.partBeam.partStatMom1.xp    = XP[0]  # numpy.random.normal(0.0,sigXp)
# 		# wfr.partBeam.partStatMom1.y     =  Y[0]  # numpy.random.normal(0.0,sigY)
# 		# wfr.partBeam.partStatMom1.yp    = YP[0]  # numpy.random.normal(0.0,sigYp)
# 		# wfr.partBeam.partStatMom1.gamma =  E[0]/electron_rest_energy_in_GeV # numpy.random.normal(electron_energy_in_GeV/electron_rest_energy_in_GeV,
# 		# 									#			  sigEperE*electron_energy_in_GeV/electron_rest_energy_in_GeV)
#
# 		print("----> Electron Coord.: x= %g x'= %g y= %g y'= %g E= %g"%(
# 			wfr.partBeam.partStatMom1.x,
# 			wfr.partBeam.partStatMom1.xp,
# 			wfr.partBeam.partStatMom1.y,
# 			wfr.partBeam.partStatMom1.yp,
# 			wfr.partBeam.partStatMom1.gamma*electron_rest_energy_in_GeV))
#
#
# 		print('- Performing Initial Electric Field calculation ... ')
# 		srwl.CalcElecFieldSR(wfr, eTraj, magFldCnt, arPrecSR)
#
# 		#
# 		# plot source
# 		#
# 		if do_plots:
# 			arI1,x,y = plot_wfr(wfr,kind='intensity',title='Source Intensity at ' + str(wfr.mesh.eStart) + ' eV',
# 					 xtitle='Horizontal Position [um]',
# 					 ytitle='Vertical Position [um]',aspect=None,show=True)
#
#
# 			# arP1,x,y = plot_wfr(wfr,kind='phase',title='Source Phase at ' + str(wfr.mesh.eStart) + ' eV',
# 			# 		 xtitle='Horizontal Position [um]',
# 			# 		 ytitle='Vertical Position [um]',show=True)
#
# 		if dump_file:
# 			filename = "tmp.h5"
# 			f = dump_open(filename)
# 			dump_wfr(f,wfr,prefix="source")
#
# 		#********************************Electrical field propagation
# 		print('- Simulating Electric Field Wavefront Propagation ... ')
#
# 		srwl.PropagElecField(wfr, optBL)
#
# 		if do_plots:
# 			arI1,x,y = plot_wfr(wfr,kind='intensity',title='Focal Intensity at ' + str(wfr.mesh.eStart) + ' eV',
# 					 xtitle='Horizontal Position [um]',
# 					 ytitle='Vertical Position [um]',show=True)
#
#
# 			# arP1,x,y = plot_wfr(wfr,kind='phase',title='Focal Phase at ' + str(wfr.mesh.eStart) + ' eV',
# 			# 		 xtitle='Horizontal Position [um]',
# 			# 		 ytitle='Vertical Position [um]',show=True)
#
# 		if dump_file:
# 			dump_wfr(f,wfr,prefix="focus")
# 			dump_close(f)
# 			print("File written to disk: %s"%(filename))
#
# 	#
# 	# start multi electron calculation
# 	#
# 	tic()
#
# 	if MultiE == 0:
# 		pass
# 	elif MultiE == 1:
# 		f = dump_open("tmp_multi1.h5")
# 		print('- Simulating Partially-Coherent Wavefront Propagation by summing-up contributions of SR from individual electrons (takes time)... ')
# 		nMacroElecAvgPerProc =  0	# number of macro-electrons / wavefront to average on worker processes before sending data to master (for parallel calculation only)
# 		nMacroElecSavePer = 100		# intermediate data saving periodicity (in macro-electrons)
# 		srCalcMeth = 1				# SR calculation method
# 		srCalcPrec = 0.01			# SR calculation rel. accuracy
# 		radStokesProp = srwl_wfr_emit_prop_multi_e_NEW(eBeam, magFldCnt, meshPartCoh, srCalcMeth, srCalcPrec,
# 													   nMacroElec, nMacroElecAvgPerProc, nMacroElecSavePer,
# 													   os.path.join(os.getcwd(), strDataFolderName, strIntPrtlChrnc),
# 													   sampFactNxNyForProp, optBL,_char=0,
# 													   filename="tmp_multi11.h5")
#
#
# 		# arxx = numpy.array(radStokesProp.arS)
# 		# arxx = arxx.reshape((4,radStokesProp.mesh.ny,radStokesProp.mesh.nx)).T
# 		# print(">>>>",arxx.shape)
# 		# print("X:",radStokesProp.mesh.xStart,radStokesProp.mesh.xFin,radStokesProp.mesh.nx)
# 		# print("Y:",radStokesProp.mesh.yStart,radStokesProp.mesh.yFin,radStokesProp.mesh.ny)
# 		# f["multielectron_stokes0"] = arxx[:,:,0]
# 		# f["multielectron_stokes1"] = arxx[:,:,1]
# 		# f["multielectron_stokes2"] = arxx[:,:,2]
# 		# f["multielectron_stokes3"] = arxx[:,:,3]
#
#
# 		f.close()
#
# 	elif MultiE == 2:
# 		f = dump_open("tmp_multi2.h5")
#
# 		INTS = numpy.zeros((nMacroElec,wfr.mesh.nx,wfr.mesh.ny))
#
# 		for i in range(nMacroElec):
#
# 			# redefine electron initial condition with random values
# 			wfr = deepcopy(wfr0)
# 			# i= 0 Electron Coord.: x= -5.31942961516259e-05 x'= -4.256161067209395e-06 y= -6.985445582705632e-07 y'= 8.357970620475086e-07 E= 6.002326228169875
# 			# i= 1 Electron Coord.: x= -2.7074235691140736e-05 x'= 5.168259996376612e-06 y= -6.825593892060933e-07 y'= -2.2658468477885813e-07 E= 6.004500995161671
# 			# i= 2 Electron Coord.: x= -4.648607227005506e-06 x'= -1.8516748893362065e-06 y= 1.208825514277266e-07 y'= -5.517433654137355e-08 E= 5.992354787190328
# 			X = [-5.31942961516259e-05,-2.7074235691140736e-05,-4.648607227005506e-06]
# 			Y =  [-6.985445582705632e-07,-6.825593892060933e-07,1.208825514277266e-07]
# 			XP =  [-4.256161067209395e-06,5.168259996376612e-06 ,-1.8516748893362065e-06]
# 			YP =  [8.357970620475086e-07,-2.2658468477885813e-07 ,-5.517433654137355e-08]
# 			E = [ 6.002326228169875,6.004500995161671,5.992354787190328]
#
# 			wfr.partBeam.partStatMom1.x     = X[i]  # numpy.random.normal(0.0,sigX)
# 			wfr.partBeam.partStatMom1.xp    = XP[i] # numpy.random.normal(0.0,sigXp)
# 			wfr.partBeam.partStatMom1.y     = Y[i] # numpy.random.normal(0.0,sigY)
# 			wfr.partBeam.partStatMom1.yp    = YP[i] # numpy.random.normal(0.0,sigYp)
# 			wfr.partBeam.partStatMom1.gamma = E[i]/electron_rest_energy_in_GeV # numpy.random.normal(electron_energy_in_GeV/electron_rest_energy_in_GeV,
# 												#			  sigEperE*electron_energy_in_GeV/electron_rest_energy_in_GeV)
#
#
# 			print("----> i= %d Electron Coord.: x= %g x'= %g y= %g y'= %g E= %g"%(i,
# 				wfr.partBeam.partStatMom1.x,
# 				wfr.partBeam.partStatMom1.xp,
# 				wfr.partBeam.partStatMom1.y,
# 				wfr.partBeam.partStatMom1.yp,
# 				wfr.partBeam.partStatMom1.gamma*electron_rest_energy_in_GeV))
#
# 			srwl.CalcElecFieldSR(wfr, eTraj, magFldCnt, arPrecSR)
# 			srwl.PropagElecField(wfr, optBL)
#
# 			ar1 = array('f', [0]*wfr.mesh.nx*wfr.mesh.ny) # "flat" 2D array to take intensity data
# 			srwl.CalcIntFromElecField(ar1, wfr, 6, 0, 3, wfr.mesh.eStart, 0, 0)
# 			arxx = numpy.array(ar1)
# 			arxx = arxx.reshape((wfr.mesh.ny,wfr.mesh.nx)).T
# 			print(">>>>",arxx.shape,INTS.shape)
# 			INTS[i,:,:] = arxx[:,:]
#
# 		f["multielectron_intensity"] = INTS
# 		f.close()
#
#
# 	print('   done')
#
# 	toc()
#
#
# 	print('\ncalculation finished\n')


if __name__ == "__main__":
    from srw_h5 import save_wfr_2_hdf5, load_h5_file_to_dictionary, dictionary_to_srw_wavefront, load_h5_file_to_srw_wavefront

    calculate_or_load = 1 # 0=calculate 1=load

    if calculate_or_load == 0:
        wfr = calculate_source(do_plots=False,dump_file=False)

        save_wfr_2_hdf5(wfr,"tmp2.h5",_complex_amplitude=True,_intensity=True,_amplitude=False,_phase=True,_overwrite=True)

        wfr_end = propagate_beamline(wfr,do_plots=False,dump_file=False)

        save_wfr_2_hdf5(wfr_end,"tmp2_end.h5",_complex_amplitude=True,_intensity=True,_amplitude=False,_phase=False,
                        _overwrite=True,_subgroupname="wfr")


    elif calculate_or_load == 1:
        wfr_loaded = load_h5_file_to_srw_wavefront("tmp2.h5","wfr")
        save_wfr_2_hdf5(wfr_loaded,"tmp2bis.h5",_complex_amplitude=True,_intensity=True,_amplitude=False,_phase=False,_overwrite=True)
        wfr_end2 = propagate_beamline(wfr_loaded,do_plots=False,dump_file=False)
        save_wfr_2_hdf5(wfr_end2,"tmp2bis_end.h5",_complex_amplitude=True,_intensity=True,_amplitude=False,_phase=True,
                        _overwrite=True,_subgroupname="wfr_end")
