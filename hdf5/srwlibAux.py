############################################################################
# SRWLibAux for Python
# Author: Rafael Celestre
# Rafael.Celestre@esrf.fr
# 09.01.2018
#############################################################################
"""
MIT License

Copyright (c) 2018 Rafael Celestre

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
#############################################################################
from __future__ import print_function #Python 2.7 compatibility
from srwlib import *
from array import *
from math import *
from copy import *
from uti_math import*
import datetime
import json
import random
import sys
import os
import traceback
import uti_math
import errno
import tempfile
import time
import shutil

try:
    import h5py
    has_h5py = True
except:
    print("h5py is not installed")
    has_h5py = False

try:
    import numpy
    has_numpy = True
except:
    print("NumPy is not installed")
    has_numpy = False
# ****************************************************************************
# ****************************************************************************
# Global Constants
# ****************************************************************************
# ****************************************************************************
_Pi = 3.14159265358979
_ElCh = 1.60217646263E-19 #1.602189246E-19 #Electron Charge [Q]
_ElMass_kg = 9.1093818872E-31 #9.10953447E-31 #Electron Mass in [kg]
_ElMass_MeV = 0.51099890221 #Electron Mass in [MeV]
_LightSp = 2.9979245812E+08 #Speed of Light [m/c]
_Light_eV_mu = 1.23984186 #Wavelength <-> Photon Energy conversion constant ([um] <-> [eV])
_PlanckConst_eVs = 4.13566766225E-15 #Planck constant in [eV*s]
# ****************************************************************************

# ****************************************************************************
# ****************************************************************************
# **********************Auxiliary data storage functions
# ****************************************************************************
# ****************************************************************************

def SRWdat_2_h5(_filename):
    """
    Auxiliary to be convert output files from SRW .dat format into a generic wavefront hdf5 generic file
    :param _filename: path to file for saving the wavefront
    """
    print(">>>> Function not implemented yet")

# TODO: change name for amplitude? remove _overwrite? remove underscore for arguments?
def save_wfr_2_hdf5(_wfr,_filename,_subgroupname="wfr",_intensity=False,_amplitude=False,_phase=False,_overwrite=True):
    """
    Auxiliary function to write wavefront data into a hdf5 generic file.
    When using the append mode to write h5 files, overwriting does not work and makes the code crash. To avoid this
    issue, try/except is used. If by any chance a file should be overwritten, it is firstly deleted and re-written.
    :param _wfr: input / output resulting Wavefront structure (instance of SRWLWfr);
    :param _filename: path to file for saving the wavefront
    :param _subgroupname: container mechanism by which HDF5 files are organised
    :param _intensity: writes intensity for sigma and pi polarisation (default=False)
    :param _amplitude: Single-Electron" Intensity - total polarisation (instance of srwl.CalcIntFromElecField)
    :param _phase: "Single-Electron" Radiation Phase - total polarisation (instance of srwl.CalcIntFromElecField)
    :param _overwrite: flag that should always be set to True to avoid infinity loop on the recursive part of the function.
    """

    try:
        if not os.path.isfile(_filename):  # if file doesn't exist, create it.
            sys.stdout.flush()
            f = h5py.File(_filename, 'w')
            # point to the default data to be plotted
            f.attrs['default']          = 'entry'
            # give the HDF5 root some more attributes
            f.attrs['file_name']        = _filename
            f.attrs['file_time']        = time.time()
            f.attrs['creator']          = 'save_wfr_2_hdf5'
            f.attrs['HDF5_Version']     = h5py.version.hdf5_version
            f.attrs['h5py_version']     = h5py.version.version
            f.close()

        # always writes complex amplitude
        # if _complex_amplitude:
        x_polarization = _SRW_2_Numpy(_wfr.arEx, _wfr.mesh.nx, _wfr.mesh.ny, _wfr.mesh.ne)   # sigma
        y_polarization = _SRW_2_Numpy(_wfr.arEy, _wfr.mesh.nx, _wfr.mesh.ny, _wfr.mesh.ne)   # pi

        e_field = numpy.concatenate((x_polarization, y_polarization), 3)

        _dump_arr_2_hdf5(e_field[0,:,:,0], "wfr_complex_amplitude_sigma", _filename, _subgroupname)
        _dump_arr_2_hdf5(e_field[0,:,:,1], "wfr_complex_amplitude_pi", _filename, _subgroupname)

        # writes now optional data blocks
        # TODO: rm amplitude? intensity and aplitude are now the same?
        if _intensity:
            # signal data
            intens = numpy.abs(e_field[0,:,:,0])**2 + numpy.abs(e_field[0,:,:,1])**2
            _dump_arr_2_hdf5(intens.T, "intensity/wfr_intensity_transposed", _filename, _subgroupname)

        if _amplitude:
            ar1 = array('f', [0] * _wfr.mesh.nx * _wfr.mesh.ny)  # "flat" 2D array to take intensity data
            srwl.CalcIntFromElecField(ar1, _wfr, 6, 0, 3, _wfr.mesh.eStart, 0, 0)
            arxx = numpy.array(ar1)
            arxx = arxx.reshape((_wfr.mesh.ny, _wfr.mesh.nx)) #.T

            _dump_arr_2_hdf5(arxx,"amplitude/wfr_amplitude_transposed", _filename, _subgroupname)

        if _phase:
            ar1 = array('d', [0] * _wfr.mesh.nx * _wfr.mesh.ny)  # "flat" 2D array to take intensity data
            srwl.CalcIntFromElecField(ar1, _wfr, 0, 4, 3, _wfr.mesh.eStart, 0, 0)
            arxx = numpy.array(ar1)
            arxx = arxx.reshape((_wfr.mesh.ny, _wfr.mesh.nx)) #.T

            _dump_arr_2_hdf5(arxx, "phase/wfr_phase_transposed", _filename, _subgroupname)


        # add mesh and SRW information
        f = h5py.File(_filename, 'a')
        f1 = f[_subgroupname]

        # point to the default data to be plotted
        f1.attrs['NX_class'] = 'NXentry'
        f1.attrs['default']          = 'intensity'

        f1["wfr_method"] = "SRW"
        f1["wfr_photon_energy"] = _wfr.mesh.eStart
        f1["wfr_radii"] =  numpy.array([_wfr.Rx,_wfr.dRx,_wfr.Ry,_wfr.dRy])
        f1["wfr_mesh"] =  numpy.array([_wfr.mesh.xStart,_wfr.mesh.xFin,_wfr.mesh.nx,_wfr.mesh.yStart,_wfr.mesh.yFin,_wfr.mesh.ny])

        # Add NX plot attribites for automatic plot with silx view
        myflags = [_intensity,_amplitude,_phase]
        mylabels = ['intensity','amplitude','phase']
        for i,label in enumerate(mylabels):
            if myflags[i]:
                f2 = f1[mylabels[i]]
                f2.attrs['NX_class'] = 'NXdata'
                f2.attrs['signal'] = 'wfr_%s_transposed'%(mylabels[i])
                f2.attrs['axes'] = [b'axis_y', b'axis_x']

                # ds = nxdata.create_dataset('image_data', data=data)
                f3 = f2["wfr_%s_transposed"%(mylabels[i])]
                f3.attrs['interpretation'] = 'image'

                # X axis data
                ds = f2.create_dataset('axis_y', data=1e6*numpy.linspace(_wfr.mesh.yStart,_wfr.mesh.yFin,_wfr.mesh.ny))
                # f1['axis1_name'] = numpy.arange(_wfr.mesh.ny)
                ds.attrs['units'] = 'microns'
                ds.attrs['long_name'] = 'Y Pixel Size (microns)'    # suggested X axis plot label
                #
                # Y axis data
                ds = f2.create_dataset('axis_x', data=1e6*numpy.linspace(_wfr.mesh.xStart,_wfr.mesh.xFin,_wfr.mesh.nx))
                ds.attrs['units'] = 'microns'
                ds.attrs['long_name'] = 'X Pixel Size (microns)'    # suggested Y axis plot label
        f.close()

    except:
        # TODO: check exit??
        # TODO: check exit??
        if _overwrite is not True:
            print(">>>> Bad input argument")
            sys.exit()
        os.remove(_filename)
        print(">>>> save_wfr_2_hdf5: file deleted %s"%_filename)

        FileName = _filename.split("/")
        # print(">>>> save_wfr_2_hdf5: %s"%_subgroupname+" in %s was deleted." %FileName[-1])
        save_wfr_2_hdf5(_wfr,_filename,_subgroupname,_intensity=_intensity,_amplitude=_amplitude,_phase=_phase,_overwrite=False)

    print(">>>> save_wfr_2_hdf5: witten/updated %s data in file: %s"%(_subgroupname,_filename))
def save_stokes_2_hdf5(_Stokes,_filename,_subgroupname="wfr",_S0=True,_S1=False,_S2=False,_S3=False,_overwrite=True):
    """
     Auxiliary function to write the Stokes parameters data into a hdf5 generic file.
    :param _Stokes: input / output resulting radiation stokes Parameters (instance of SRWLStokes)
    :param _filename: path to file for saving the wavefront
    :param _subgroupname: container mechanism by which HDF5 files are organised
    :param _S0: I = P_0 + P_90 = <Ex^2 + Ey^2>
    :param _S1: Q = P_0 - P_90 = <Ex^2 - Ey^2>
    :param _S2: U = P_45 + P_135 = <2Ex*Ey*cos(delta)>
    :param _S3: V = P_r_circular + P_l_circular = <2Ex*Ey*sin(delta)>
    :param _overwrite: flag that should always be set to True to avoid infinity loop on the recursive part of the function.
    """
    try:
        if not os.path.isfile(_filename):  # if file doesn't exist, create it.
            sys.stdout.flush()
            f = h5py.File(_filename, 'w')
            f.close()

        arxx = numpy.array(_Stokes.arS)
        arxx = arxx.reshape((4, _Stokes.mesh.ny, _Stokes.mesh.nx)).T

        if _S0:
            _dump_arr_2_hdf5(arxx[:, :, 0], "Stokes_S0", _filename, _subgroupname)
        if _S1:
            _dump_arr_2_hdf5(arxx[:, :, 1], "Stokes_S1", _filename, _subgroupname)
        if _S2:
            _dump_arr_2_hdf5(arxx[:, :, 2], "Stokes_S2", _filename, _subgroupname)
        if _S3:
            _dump_arr_2_hdf5(arxx[:, :, 3], "Stokes_S3", _filename, _subgroupname)

        f = h5py.File(_filename, 'a')
        f1 = f[_subgroupname]
        f1["Stokes_method"] = "SRW"
        f1["Stokes_photon_energy"] = _Stokes.mesh.eStart
        f1["Stokes_mesh"] = numpy.array([_Stokes.mesh.xStart, _Stokes.mesh.xFin, _Stokes.mesh.nx, _Stokes.mesh.yStart, _Stokes.mesh.yFin, _Stokes.mesh.ny])
        f.close()
    except:
        if _overwrite is not True:
            print(">>>> Bad input argument")
            sys.exit()
        os.remove(_filename)

        FileName = _filename.split("/")
        print(">>>> save_stokes_2_hdf5 warning: %s"%_subgroupname+" in %s was deleted." %FileName[-1])
        save_stokes_2_hdf5(_Stokes,_filename,_subgroupname,_S0,_S1,_S2,_S3,_overwrite = False)

def _SRW_2_Numpy(_srw_array, _nx, _ny, _ne):
    """
    Converts an SRW array to a numpy.array.
    :param _srw_array: SRW array
    :param _nx: numbers of points vs horizontal positions
    :param _ny: numbers of points vs vertical positions
    :param _ne: numbers of points vs photon energy
    :return: 4D numpy array: [energy, horizontal, vertical, polarisation={0:horizontal, 1: vertical}]
    """
    re = numpy.array(_srw_array[::2], dtype=numpy.float)
    im = numpy.array(_srw_array[1::2], dtype=numpy.float)

    e = re + 1j * im
    e = e.reshape((_ny,_nx,_ne,1))
    e = e.swapaxes(0, 2)

    return e.copy()

def _dump_arr_2_hdf5(_arr,_calculation, _filename, _subgroupname):
    """
    Auxiliary routine to save_wfr_2_hdf5()
    :param _arr: (usually 2D) array to be saved on the hdf5 file inside the _subgroupname
    :param _filename: path to file for saving the wavefront
    :param _subgroupname: container mechanism by which HDF5 files are organised
    """
    sys.stdout.flush()
    f = h5py.File(_filename, 'a')
    try:
        f1 = f.create_group(_subgroupname)
    except:
        f1 = f[_subgroupname]
    # f1[_calculation] = _arr
    fdata = f1.create_dataset(_calculation,data=_arr)
    f.close()

    return fdata

# ****************************************************************************
# ****************************************************************************
# **********************Auxiliary generic functions
# ****************************************************************************
# ****************************************************************************

def wrf_caustic(_di,_df,_pts,_wfr,_optBL,_filename, _inPol=6, _inIntType = 0, _inDepType = 7, _inE = None, _inX = 0, _inY = 0):
    """
    Auxiliary beam propagation function for simulation the beam caustics (beam evolution along the optical axis). For
    the best performance of the function some precautions must be taken:
        - the last optical element of the beamline (container of optical elements - instance of SRWLOptC) should be a drift space
        - the lower range (_di) must not exceed the distance given by the abovementioned drift space
        - proceed with caution if not using a (Coherent) Gaussian (Radiation) Beam (instance of SRWLGsnBm)
        - function is implemented for monochromatic Wavefront structure (instance of SRWLWfr)
    Files are saved in hdf5 format.
    :param _di: initial relative position (to the last drift in _optBL) along the optical axis
    :param _df: final relative position (to the last drift in _optBL) along the optical axis (_df > _di)
    :param _pts: number of equally spaced points for caustic calculations
    :param _wfr: path to file for saving the beam caustics
    :param _optBL: optical beamline (container) to propagate the radiation through (SRWLOptC type)
    :param _filename: path to file for saving the wavefront
    :param _inPol: input switch specifying polarization component to be extracted:
               =0 -Linear Horizontal;
               =1 -Linear Vertical;
               =2 -Linear 45 degrees;
               =3 -Linear 135 degrees;
               =4 -Circular Right;
               =5 -Circular Left;
               =6 -Total
    :param _inIntType:input switch specifying "type" of a characteristic to be extracted:
               =0 -"Single-Electron" Intensity;
               =1 -"Multi-Electron" Intensity;
               =2 -"Single-Electron" Flux;
               =3 -"Multi-Electron" Flux;
               =4 -"Single-Electron" Radiation Phase;
               =5 -Re(E): Real part of Single-Electron Electric Field;
               =6 -Im(E): Imaginary part of Single-Electron Electric Field;
               =7 -"Single-Electron" Intensity, integrated over Time or Photon Energy (i.e. Fluence)
    :param _inDepType: input switch specifying type of dependence to be extracted:
               =1 -vs x (horizontal position or angle);
               =2 -vs y (vertical position or angle);
               =3 -vs x&y (horizontal and vertical positions or angles);
               =7 -vs x (horizontal position or angle) and vs y (vertical position or angle);
               =8 -vs x (horizontal position or angle) - averaged/collapsed;
               =9 -vs y (vertical position or angle) - averaged/collapsed;
               =10 -vs x (horizontal position or angle) and vs y (vertical position or angle) -  averaged/collapsed;
    :param _inE: input photon energy [eV] or time [s] to keep fixed (to be taken into account for dependences vs x, y, x&y)
    :param _inX: input horizontal position [m] to keep fixed (to be taken into account for dependences vs e, y, e&y)
    :param _inY: input vertical position [m] to keep fixed (to be taken into account for dependences vs e, x, e&x)
    """
    print()
    if _inE is None:
        _inE = _wfr.mesh.eStart

    delta_Z = numpy.linspace(_di, _df, num=_pts)  #

    z0 = _optBL.arOpt[-1].L

    if (z0+_df) < 0:
        print(">>>> Error: trying to start calculation before the last optical element. Check simulation range.")
        sys.exit()

    for k in range(_pts):
        print(">>>> Caustics: point %d"%(k+1)+" out of %d"%_pts)

        wfrp = deepcopy(_wfr)
        position = z0 + delta_Z[k]

        _optBL.arOpt[-1] = SRWLOptD(position)

        srwl.PropagElecField(wfrp,_optBL)

        # -vs x (horizontal position or angle);
        if _inDepType == 1:
            if k == 0:
                ar_xi = array('f', [0] * _pts)
                ar_xf = array('f', [0] * _pts)
                Cst_vs_x = NaN((wfrp.mesh.nx, _pts), 'f')

            arI = array('f', [0] * wfrp.mesh.nx)

            srwl.CalcIntFromElecField(arI, wfrp, _inPol, _inIntType, 1, _inE, _inX, _inY)

            arI = numpy.array(arI)
            Cst_vs_x[:, k] = arI

            ar_xi[k] = wfrp.mesh.xStart
            ar_xf[k] = wfrp.mesh.xFin

        # -vs y (vertical position or angle);
        if _inDepType == 2:
            if k == 0:
                ar_yi = array('f', [0] * _pts)
                ar_yf = array('f', [0] * _pts)
                Cst_vs_y = NaN((wfrp.mesh.ny,_pts), 'f')

            arI = array('f', [0] * wfrp.mesh.ny)

            srwl.CalcIntFromElecField(arI, wfrp, _inPol, _inIntType, 2, _inE, _inX, _inY)
            arI = numpy.array(arI)
            Cst_vs_y[:,k] = arI

            ar_yi[k] = wfrp.mesh.yStart
            ar_yf[k] = wfrp.mesh.yFin

        # -vs x&y (horizontal and vertical positions or angles);
        if _inDepType == 3:
            subgroupname = "wfr_%d"%k

            arI = array('f', [0] * wfrp.mesh.nx * wfrp.mesh.ny)

            srwl.CalcIntFromElecField(arI, wfrp, _inPol, _inIntType, 3, _inE, _inX, _inY)

            arI = numpy.array(arI)
            arI = arI.reshape((wfrp.mesh.ny, wfrp.mesh.nx)).T

            wfr_mesh = [wfrp.mesh.eStart, wfrp.mesh.xStart, wfrp.mesh.xFin, wfrp.mesh.nx, wfrp.mesh.yStart,wfrp.mesh.yFin, wfrp.mesh.ny,_di,_df,_pts]
            if k == 0:
                _save_caustic_2_hdf5(arI,wfr_mesh,_filename,subgroupname,_overwrite=False)
            else:
                _save_caustic_2_hdf5(arI, wfr_mesh, _filename, subgroupname)

        # -vs x (horizontal position or angle) & -vs y (vertical position or angle) - DEFAULT;
        if _inDepType == 7:
            # -vs x
            if k == 0:
                ar_xi = array('f', [0] * _pts)
                ar_xf = array('f', [0] * _pts)
                Cst_vs_x = NaN((wfrp.mesh.nx,_pts), 'f')

            arI = array('f', [0] * wfrp.mesh.nx)

            srwl.CalcIntFromElecField(arI, wfrp, _inPol, _inIntType, 1, _inE, _inX, _inY)

            arI = numpy.array(arI)
            Cst_vs_x[:,k] = arI

            ar_xi[k] = wfrp.mesh.xStart
            ar_xf[k] = wfrp.mesh.xFin

            # -vs y
            if k == 0:
                ar_yi = array('f', [0] * _pts)
                ar_yf = array('f', [0] * _pts)
                Cst_vs_y = NaN((wfrp.mesh.ny,_pts), 'f')

            arI = array('f', [0] * wfrp.mesh.ny)

            srwl.CalcIntFromElecField(arI, wfrp, _inPol, _inIntType, 2, _inE, _inX, _inY)

            arI = numpy.array(arI)
            Cst_vs_y[:,k] = arI

            ar_yi[k] = wfrp.mesh.yStart
            ar_yf[k] = wfrp.mesh.yFin

        # -vs x (horizontal position or angle) - COLLAPSED;
        if _inDepType == 8:
            if k == 0:
                ar_xi = array('f', [0] * _pts)
                ar_xf = array('f', [0] * _pts)
                Cst_vs_x = NaN((wfrp.mesh.nx, _pts), 'f')

            arI = array('f', [0] * wfrp.mesh.nx * wfrp.mesh.ny)

            srwl.CalcIntFromElecField(arI, wfrp, _inPol, _inIntType, 3, _inE, _inX, _inY)

            arI = numpy.array(arI)
            arI = arI.reshape((wfrp.mesh.ny, wfrp.mesh.nx)).T

            for x in range(wfrp.mesh.nx):
                Cst_vs_x[x,k] = numpy.sum(arI[x,:])/wfrp.mesh.ny

            ar_xi[k] = wfrp.mesh.xStart
            ar_xf[k] = wfrp.mesh.xFin

        # -vs y (vertical position or angle) - COLLAPSED;
        if _inDepType == 9:
            if k == 0:
                ar_yi = array('f', [0] * _pts)
                ar_yf = array('f', [0] * _pts)
                Cst_vs_y = NaN((wfrp.mesh.ny,_pts), 'f')

            arI = array('f', [0] * wfrp.mesh.nx * wfrp.mesh.ny)

            srwl.CalcIntFromElecField(arI, wfrp, _inPol, _inIntType, 3, _inE, _inX, _inY)

            arI = numpy.array(arI)
            arI = arI.reshape((wfrp.mesh.ny, wfrp.mesh.nx)).T

            for y in range(wfrp.mesh.ny):
                Cst_vs_y[y,k] = numpy.sum(arI[:,y])/wfrp.mesh.nx

            ar_yi[k] = wfrp.mesh.yStart
            ar_yf[k] = wfrp.mesh.yFin

        if _inDepType == 10:                 # -vs x (horizontal position or angle) & -vs y (vertical position or angle) - COLLAPSED;
            if k == 0:
                ar_xi = array('f', [0] * _pts)
                ar_xf = array('f', [0] * _pts)
                ar_yi = array('f', [0] * _pts)
                ar_yf = array('f', [0] * _pts)

                Cst_vs_x = NaN((wfrp.mesh.nx,_pts), 'f')
                Cst_vs_y = NaN((wfrp.mesh.ny,_pts), 'f')

            arI = array('f', [0]*wfrp.mesh.nx*wfrp.mesh.ny)

            srwl.CalcIntFromElecField(arI, wfrp, _inPol, _inIntType, 3, _inE, _inX, _inY)

            arI = numpy.array(arI)
            arI = arI.reshape((wfrp.mesh.ny, wfrp.mesh.nx)).T

            for x in range(wfrp.mesh.nx):
                Cst_vs_x[x,k] = numpy.sum(arI[x,:])/wfrp.mesh.ny

            for y in range(wfrp.mesh.ny):
                Cst_vs_y[y,k] = numpy.sum(arI[:,y])/wfrp.mesh.nx

            ar_xi[k] = wfrp.mesh.xStart
            ar_xf[k] = wfrp.mesh.xFin

            ar_yi[k] = wfrp.mesh.yStart
            ar_yf[k] = wfrp.mesh.yFin

    print("\n>>>> Caustics: post processing. It takes time.")

    # -vs x (horizontal position or angle)
    if (_inDepType == 1) or (_inDepType == 8):
        X = numpy.linspace(numpy.amax(ar_xi),numpy.amin(ar_xf),wfrp.mesh.nx)

        Cst_vs_xn = NaN((wfrp.mesh.nx, _pts), 'f')

        for k in range(_pts):
            ar = numpy.linspace(ar_xi[k], ar_xf[k], wfrp.mesh.nx)

            for x in range(wfrp.mesh.nx):
                Cst_vs_xn[x,k] = interp_1d_var(X[x], ar, Cst_vs_x[:,k], _ord=3)

        wfr_mesh = [wfrp.mesh.eStart, numpy.amax(ar_xi), numpy.amin(ar_xf), wfrp.mesh.nx, _di, _df, _pts]

        _save_caustic_2_hdf5(Cst_vs_xn, wfr_mesh, _filename, 'Horizontal',_overwrite=False)

    # -vs y (vertical position or angle)
    if (_inDepType == 2) or (_inDepType == 9):
        Y = numpy.linspace(numpy.amax(ar_yi),numpy.amin(ar_yf),wfrp.mesh.ny)

        Cst_vs_yn = NaN((wfrp.mesh.ny, _pts), 'f')

        for k in range(_pts):
            ar = numpy.linspace(ar_yi[k], ar_yf[k], wfrp.mesh.ny)

            for y in range(wfrp.mesh.ny):
                Cst_vs_yn[y,k] = interp_1d_var(Y[y], ar, Cst_vs_y[:,k], _ord=3)

        wfr_mesh = [wfrp.mesh.eStart, numpy.amax(ar_yi), numpy.amin(ar_yf), wfrp.mesh.ny, _di, _df, _pts]

        _save_caustic_2_hdf5(Cst_vs_yn, wfr_mesh, _filename, 'Vertical',_overwrite=False)

    # -vs x (horizontal position or angle) & -vs y (vertical position or angle)
    if (_inDepType == 7) or (_inDepType == 10):
        X = numpy.linspace(numpy.amax(ar_xi),numpy.amin(ar_xf),wfrp.mesh.nx)
        Y = numpy.linspace(numpy.amax(ar_yi),numpy.amin(ar_yf),wfrp.mesh.ny)

        Cst_vs_xn = NaN((wfrp.mesh.nx, _pts), 'f')
        Cst_vs_yn = NaN((wfrp.mesh.ny, _pts), 'f')

        for k in range(_pts):
            ar = numpy.linspace(ar_xi[k], ar_xf[k], wfrp.mesh.nx)
            for x in range(wfrp.mesh.nx):
                Cst_vs_xn[x,k] = interp_1d_var(X[x], ar, Cst_vs_x[:,k], _ord=3)

            ar = numpy.linspace(ar_yi[k], ar_yf[k], wfrp.mesh.ny)
            for y in range(wfrp.mesh.ny):
                Cst_vs_yn[y,k] = interp_1d_var(Y[y], ar, Cst_vs_y[:,k], _ord=3)

        wfr_mesh = [wfrp.mesh.eStart,numpy.amax(ar_xi),numpy.amin(ar_xf),wfrp.mesh.nx,_di,_df,_pts]
        _save_caustic_2_hdf5(Cst_vs_xn,wfr_mesh,_filename,'Horizontal',_overwrite=False)

        wfr_mesh = [wfrp.mesh.eStart,numpy.amax(ar_yi),numpy.amin(ar_yf),wfrp.mesh.ny,_di,_df,_pts]
        _save_caustic_2_hdf5(Cst_vs_yn,wfr_mesh,_filename,'Vertical',_overwrite=True)

    print("\n>>>> Caustics: calculation finished. Data saved to file.")

def _save_caustic_2_hdf5(_arr,_wfr_mesh, _filename, _subgroupname,_overwrite=True):
    """
    Auxiliary routine to save the beam caustics
    :param _arr: array to be saved on the hdf5 file inside the _subgroupname
    :param _wfr_mesh: caustic mesh
    :param _filename: path to file for saving the wavefront
    :param _subgroupname: container mechanism by which HDF5 files are organised
    :param _overwrite: flat to avid conflict when overwriting hdf5 in the append (a) mode

    """
    try:
        if _overwrite is not True:
            os.remove(_filename)

        sys.stdout.flush()

        f = h5py.File(_filename, 'a')

        try:
            f1 = f.create_group(_subgroupname)
        except:
            f1 = f[_subgroupname]

        f1["wfr_method"] = "SRW"
        f1["wfr_photon_energy"] = _wfr_mesh[0]
        f1["wfr_radii"] = numpy.array([0, 0, 0, 0])
        f1["wfr_mesh"] = numpy.array(_wfr_mesh[1::])
        f1["wfr_caustic"] = _arr
        f.close()

    except:
        if _overwrite is not True:
            print(">>>> Bad input argument")
            sys.exit()

        _save_caustic_2_hdf5(_arr,_wfr_mesh, _filename, _subgroupname,_overwrite=False)

# ****************************************************************************
# ****************************************************************************
# **********************Auxiliary optical functions
# ****************************************************************************
# ****************************************************************************

# def srwl_opt_setup_speckle_membrane

def srwl_opt_setup_CRL_error(_file_path,_delta,_atten_len,_amp_coef=1, _apert_h=None, _apert_v=None,_xc=0,_yc=0,_extTr=1):
    """
    Setup Transmission type Optical Element which simulates Compound Refractive Lens (CRL) figure errors in [m]
    (format is defined in srwl_uti_save_intens_ascii).
    :param _file_path: Figure error array data from an ASCII file (format is defined in srwl_uti_save_intens_ascii)
    :param _delta: refractive index decrement (can be one number of array vs photon energy)
    :param _atten_len: attenuation length [m] (can be one number of array vs photon energy)
    :param _apert_h: horizontal aperture size [m]
    :param _apert_v: vertical aperture size [m]
    :param _xc: horizontal coordinate of center [m]
    :param _yc: vertical coordinate of center [m]
    :param _extTr: transmission outside the grid/mesh is zero (0), or it is same as on boundary (1)
    :return: transmission (SRWLOptT) type optical element which simulates CRL figure errors
    """
    print('>>>> Loading figure errors and generating transmission and phase array...')
    arPrecCRLerr = [0] * 9
    ThicknessError, arPrecCRLerr = srwl_uti_read_intens_ascii(_file_path)
    if (_apert_h is None):
        RangeX = arPrecCRLerr.xFin - arPrecCRLerr.xStart
    else:
        RangeX = _apert_h
    if (_apert_v is None):
        RangeY = arPrecCRLerr.yFin - arPrecCRLerr.yStart
    else:
        RangeY = _apert_v
    # for future versions, allow to crop/resample the grid RC 16.11.17
    Npx = arPrecCRLerr.nx
    Npy = arPrecCRLerr.ny

    elmts = 2 * Npx * Npy
    arTr = array('d', [0] * elmts)
    c1 = 0
    c2 = 0
    for n in range(0, elmts):
        if (n % 2) == 0:
            arTr[n] = exp(-_amp_coef * ThicknessError[c1] / _atten_len)
            c1 += 1
        else:
            arTr[n] = -_amp_coef * ThicknessError[c2] * _delta
            c2 += 1

    oeCRLerr = SRWLOptT(Npx, Npy, RangeX, RangeY, arTr, _extTr, _x=_xc,_y=_yc)

    input_parms = {
        "type": "crl_figure_error",
        "refractiveIndex": _delta,
        "attenuationLength": _atten_len,
        "horizontalApertureSize": _apert_h,
        "verticalApertureSize": _apert_v,
        "nheightAmplification": _amp_coef,
        "horizontalCenterCoordinate": _xc,
        "verticalCenterCoordinate": _yc,
        "horizontalPoints": Npx,
        "verticalPoints": Npy,
    }

    oeCRLerr.input_parms = input_parms

    return oeCRLerr

def index_of_refraction(_material,_E,_file_path):
    """
    For a given energy range (1000 eV < E < 30,000 eV), this function returns an array composed of the complex index of
    refraction and the attenuation length for a given material from the database. Existing materials are: Al, Be, C,
    Diamond, Ni, Si and SU-8 (SU8 or PMMA).
    :param _material: string (not case sensitive). Existing materials are: Al, Be, C, Diamond, Ni, Si and SU-8 (SU8 or PMMA).
    :param _E: fundamental photon energy [eV]
    :param _file_path: folder where the the .dat files for different materials are saved
    :return complex_index_of_refraction: delta, beta, atten_len
    """
    if (_E>30000) or (_E<1000):
        print(">>>> Error: energy out of range. Energies must be in the range 1000 eV < E < 30,000 eV\n")
        sys.exit()
    try:
        Wavelength = srwl_uti_ph_en_conv(_E, 'eV', 'm')
        if _material.lower() == "al":
            Mat_dir = os.path.join(_file_path,"Al.dat")
        elif _material.lower() == "be":
            Mat_dir = os.path.join(_file_path,"Be.dat")
        elif _material.lower() == "c":
            Mat_dir = os.path.join(_file_path,"C.dat")
        elif _material.lower() == "diamond":
            Mat_dir = os.path.join(_file_path,"Diamond.dat")
        elif _material.lower() == "ni":
            Mat_dir = os.path.join(_file_path,"Ni.dat")
        elif _material.lower() == "si":
            Mat_dir = os.path.join(_file_path,"Si.dat")
        elif (_material.lower() == "su8") or (_material.lower() == "su-8") or (_material.lower() == "pmma"):
            Mat_dir = os.path.join(_file_path,"Su-8.dat")

        Material = srwl_uti_read_data_cols(Mat_dir, " ", 0, -1, 2)
        delta = interp_1d_var(_E, Material[0], Material[1])
        beta  = interp_1d_var(_E, Material[0], Material[2])
        atten_len = Wavelength / (4 * pi * beta)
        complex_index_of_refraction = [delta, beta, atten_len]
        return complex_index_of_refraction

    except:
        print(">>>> Error: unknown material in the database. Existing materials are: Al, Be, C, Diamond, Ni, Si and SU-8 (SU8 or PMMA).\n")
        sys.exit()

# ****************************************************************************
# ****************************************************************************
# **********************Auxiliary undulator functions
# ****************************************************************************
# ****************************************************************************

def find_K_from_E(_E,_eBeamGamma,_undPer,_n=1,_Kmin=0.2,_Kmax=10,_nmax=15):
    """
    Auxiliary function to help determining an appropriate undulator deflection value for a given X-ray energy
    :param _E: fundamental photon energy [eV]
    :param _eBeamGamma: Lorentz factor (electron energy [GeV] * 1957)
    :param _undPer: period length [m]
    :param _n: odd harmonic number
    :param _Kmin: minumum deflection parameter
    :param _Kmax: maximum deflection parameter
    :param _nmax: highest odd harmonic to be considered
    :return K: deflection parameter
    """
    Wavelength = srwl_uti_ph_en_conv(_E,'eV','m')
    try:
        K = sqrt(2 * (2 * _n * Wavelength * _eBeamGamma ** 2 / _undPer - 1))
        if (K >= _Kmin) and (K <= _Kmax):
            return K
        else:
            GenerateError = 1/0.
    except:
        i = 0
        FindK = 0
        while FindK==0:
            h_n = 2*i+1     #harmonic number
            if h_n>_nmax:
                print(">>>> Error: The chosen energy cannot be reached. Check undulator parameters. \n")
                sys.exit()
            K = sqrt(2 * (2 * h_n * Wavelength * _eBeamGamma ** 2 / _undPer - 1))
            if (K>=_Kmin) and (K<=_Kmax):
                FindK = 1
            i = i+1
        if h_n == _n:
            return K
        else:
            print(">>>> Warning: The chosen energy cannot be reached at the current harmonic number n = "+str(_n)+". Try using the harmonic n = "+str(h_n)+" instead. \n")
            return K

def find_B_from_E(_E, _eBeamGamma, _undPer, _n=1, _Bmin=0.1, _Bmax=10, _nmax=15):
    """
    Auxiliary function to help determining an appropriate undulator magnetic field value for a given X-ray energy
    :param _E: fundamental photon energy [eV]
    :param _eBeamGamma: Lorentz factor (electron energy [GeV] * 1957)
    :param _undPer: period length [m]
    :param _n: odd harmonic number
    :param _Bmin: minimum magnetic field amplitude [T]
    :param _Bmax: maximum magnetic field amplitude [T]
    :param _nmax: highest odd harmonic to be considered
    :return B: magnetic field amplitude [T]
    """
    _Kmin = _Bmin*_undPer*93.3728962
    _Kmax = _Bmax*_undPer*93.3728962
    K = find_K_from_E(_E, _eBeamGamma, _undPer, _n, _Kmin, _Kmax, _nmax)
    B = K / (_undPer * 93.3728962)
    return B

# ****************************************************************************
# ****************************************************************************
# **********************Auxiliary mathematical functions
# ****************************************************************************
# ****************************************************************************

def NaN(_shape, _dtype=float):
    """Initialising numpy matrix to NaN"""
    NaN_matrix = numpy.empty(_shape, _dtype)
    NaN_matrix.fill(numpy.nan)
    return NaN_matrix

# ****************************************************************************
# ****************************************************************************
# **********************Auxiliary time functions
# ****************************************************************************
# ****************************************************************************

def tic():
    """TicToc function to calculate the elapsed execution time of a given code - part 1/2"""
    global startTime
    startTime = time.time()

def toc():
    """TicToc function to calculate the elapsed execution time of a given code - part 2/2"""
    print('')
    if 'startTime' in globals():
        deltaT = time.time() - startTime
        hours, minutes = divmod(deltaT, 3600)
        minutes, seconds = divmod(minutes, 60)
        print(">>>> Elapsed time: " + str(int(hours)) + "h " + str(int(minutes)) + "min " + str(seconds) + "s ")
    else:
        print(">>>> Warning: start time not set.")

def TxtLog():
    """Prints a time stamp on the terminal. Useful for long simulations"""
    StrLog = "Time stamp: " + str(datetime.now().hour) + 'h' + str(datetime.now().minute) + 'min' + str(
        datetime.now().second) + 's\t' + str(datetime.now().day) + '/' + str(datetime.now().month) + '/' + str(
        datetime.now().year) + ' (DD/MM/YYYY)\n'
    print(StrLog)

# ****************************************************************************
# ****************************************************************************
# **********************Imports from other codes
# ****************************************************************************
# ****************************************************************************

"""The following lines of code are adaptation from pieces of coding by other authors. Before each segment, there is a
short description of the functionality, whom the code was copied from, who adapted it and when the inclusion to srwlibAux
was done."""
#############################################################################
# Script used for obtaining the mutual coherence plot from the SRW
# Adapted from O. Chubar IgorPro script - private correspondence
# Authors: Luca Rebuffi, Manuel Sanchez del Rio (Python version)
# Adapted by: Rafael Celestre
# 09.01.2018
#############################################################################

def _file_load(_fname, _read_labels=1):
    nLinesHead = 11
    hlp = []

    with open(_fname, 'r') as f:
        for i in range(nLinesHead):
            hlp.append(f.readline())

    ne, nx, ny = [int(hlp[i].replace('#', '').split()[0]) for i in [3, 6, 9]]
    ns = 1
    testStr = hlp[nLinesHead - 1]
    if testStr[0] == '#':
        ns = int(testStr.replace('#', '').split()[0])

    e0, e1, x0, x1, y0, y1 = [float(hlp[i].replace('#', '').split()[0]) for i in [1, 2, 4, 5, 7, 8]]

    data = numpy.squeeze(numpy.loadtxt(_fname, dtype=numpy.float64))  # get data from file (C-aligned flat)

    allrange = e0, e1, ne, x0, x1, nx, y0, y1, ny

    arLabels = ['Photon Energy', 'Horizontal Position', 'Vertical Position', 'Intensity']
    arUnits = ['eV', 'm', 'm', 'ph/s/.1%bw/mm^2']

    if _read_labels:

        arTokens = hlp[0].split(' [')
        arLabels[3] = arTokens[0].replace('#', '')
        arUnits[3] = '';
        if len(arTokens) > 1:
            arUnits[3] = arTokens[1].split('] ')[0]

        for i in range(3):
            arTokens = hlp[i * 3 + 1].split()
            nTokens = len(arTokens)
            nTokensLabel = nTokens - 3
            nTokensLabel_mi_1 = nTokensLabel - 1
            strLabel = ''
            for j in range(nTokensLabel):
                strLabel += arTokens[j + 2]
                if j < nTokensLabel_mi_1: strLabel += ' '
            arLabels[i] = strLabel
            arUnits[i] = arTokens[nTokens - 1].replace('[', '').replace(']', '')

    return data, None, allrange, arLabels, arUnits

def _loadNumpyFormatCoh(_filename):
    data, dump, allrange, arLabels, arUnits = _file_load(_filename)

    dim_x = allrange[5]
    dim_y = allrange[8]

    dim = 1
    if dim_x > 1:
        dim = dim_x
    elif dim_y > 1:
        dim = dim_y

    np_array = data.reshape((dim, dim))
    np_array = np_array.transpose()

    if dim_x > 1:
        coordinates = numpy.linspace(allrange[3], allrange[4], dim_x)
        conj_coordinates = numpy.linspace(allrange[3], allrange[4], dim_x)
    elif dim_y > 1:
        coordinates = numpy.linspace(allrange[6], allrange[7], dim_y)
        conj_coordinates = numpy.linspace(allrange[6], allrange[7], dim_y)
    else:
        coordinates = None
        conj_coordinates = None

    return coordinates, conj_coordinates, np_array, allrange

def DegreeOfTransverseCoherence(_filename,_set_extrapolated_to_zero=True):
    """
    Converts the output files originated by srwl_wfr_emit_prop_multi_e from Mutual Intensity (Cross Spectral Density -
    CSD) to a Degree of Transverse Coherence (DoTC) file saving it in hdf5 generic file.
    :param _filename: Multual intensity array data from an ASCII file (format is defined in srwl_uti_save_intens_ascii
    :param _set_extrapolated_to_zero: zero padding of the matrix when applying rotation to it when set to TRUE
    """
    from scipy.interpolate import RectBivariateSpline
    FileName = _filename.split("/")
    print(">>>> Calculating the degree of transverse coherence: %s"%FileName[-1])

    coor, coor_conj, mutual_intensity, wfr_mesh = _loadNumpyFormatCoh(_filename)

    file_h5 = _filename.replace(".dat",".h5")

    f = h5py.File(file_h5, 'w')
    f1 = f.create_group("Cross_Spectral_Density")
    f1["CSD_method"] = "SRW"
    f1["CSD_photon_energy"] = wfr_mesh[0]
    f1["CSD_photon_mesh"] = numpy.array([wfr_mesh[3::]])
    f1["CSD"] = mutual_intensity
    if wfr_mesh[5] == 1:
        f1["CSD_direction"] = "vertical"
    else:
        f1["CSD_direction"] = "horizontal"

    interpolator0 = RectBivariateSpline(coor, coor_conj, mutual_intensity, bbox=[None, None, None, None], kx=3, ky=3,s=0)

    X = numpy.outer(coor, numpy.ones_like(coor_conj))
    Y = numpy.outer(numpy.ones_like(coor), coor_conj)

    nmResDegCoh_z = numpy.abs(interpolator0(X + Y, X - Y, grid=False)) / \
                    numpy.sqrt(numpy.abs(interpolator0(X + Y, X + Y, grid=False))) / \
                    numpy.sqrt(numpy.abs(interpolator0(X - Y, X - Y, grid=False)))

    if _set_extrapolated_to_zero:
        nx, ny = nmResDegCoh_z.shape

        idx = numpy.outer(numpy.arange(nx), numpy.ones((ny)))
        idy = numpy.outer(numpy.ones((nx)), numpy.arange(ny))

        mask = numpy.ones_like(idx)

        bad = numpy.where(idy < 1. * (idx - nx / 2) * ny / nx)
        mask[bad] = 0

        bad = numpy.where(idy > ny - 1. * (idx - nx / 2) * ny / nx)
        mask[bad] = 0

        bad = numpy.where(idy < 0.5 * ny - 1. * idx * ny / nx)
        mask[bad] = 0

        bad = numpy.where(idy > 0.5 * ny + 1. * idx * ny / nx)
        mask[bad] = 0

        nmResDegCoh_z *= mask

    if _filename is not None:

        f1 = f.create_group("Degree_of_Transverse_Coherence")
        f1["DoTC_method"] = "SRW"
        f1["DoTC_photon_energy"] = wfr_mesh[0]
        f1["DoTC_photon_mesh"] = numpy.array([wfr_mesh[3::]])
        f1["DoTC"] = nmResDegCoh_z
        if wfr_mesh[5] == 1:
            f1["DoTC_direction"] = "vertical"
        else:
            f1["DoTC_direction"] = "horizontal"
        f.close()

        FileName = file_h5.split("/")
        print(">>>> File %s written to disk."%FileName[-1])
