############################################################################
# SRWLibAux for Python
# Author: Rafael Celestre, X-ray Optics Group, ESRF
# Rafael.Celestre@esrf.fr
# 31.01.2018
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
from uti_math import*
from array import *
import time
import sys
import os

from NumpyToSRW import numpyArrayToSRWArray

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
# **********************Auxiliary data storage functions
# ****************************************************************************
# ****************************************************************************

def save_wfr_2_hdf5(_wfr,_filename,_subgroupname="wfr",_complex_amplitude=True,_intensity=False,_amplitude=False,_phase=False,_overwrite=True):
    """
    Auxiliary function to write wavefront data into a hdf5 generic file.
    When using the append mode to write h5 files, overwriting does not work and makes the code crash. To avoid this
    issue, try/except is used. If by any chance a file should be overwritten, it is firstly deleted and re-written.
    :param _wfr: input / output resulting Wavefront structure (instance of SRWLWfr);
    :param _filename: path to file for saving the wavefront
    :param _subgroupname: container mechanism by which HDF5 files are organised
    :param _complex_amplitude: complex electric field for sigma and pi polarisation
    :param _intensity: Single-Electron" Intensity - total polarisation (instance of srwl.CalcIntFromElecField)
    :param _amplitude: S0 Stokes parameter. I = P_0 + P_90 = <Ex^2 + Ey^2>.
    :param _phase: "Single-Electron" Radiation Phase - total polarisation (instance of srwl.CalcIntFromElecField)
    :param _overwrite: flag that should always be set to True to avoid infinity loop on the recursive part of the function.
    """

    if (os.path.isfile(_filename)) and (_overwrite==True):
        os.remove(_filename)
        FileName = _filename.split("/")
        print(">>>> save_wfr_2_hdf5: file deleted %s"%FileName[-1])

    if not os.path.isfile(_filename):  # if file doesn't exist, create it.
        sys.stdout.flush()
        f = h5py.File(_filename, 'w')
        # points to the default data to be plotted
        f.attrs['default']          = 'entry'
        # give the HDF5 root some more attributes
        f.attrs['file_name']        = _filename
        f.attrs['file_time']        = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        f.attrs['creator']          = 'save_wfr_2_hdf5'
        f.attrs['code']             = 'SRW'
        f.attrs['HDF5_Version']     = h5py.version.hdf5_version
        f.attrs['h5py_version']     = h5py.version.version
        f.close()

    # TODO: delete
    if _amplitude:
        print(">>>>>>>> writing amplitude...")
        ar1 = array('f', [0] * _wfr.mesh.nx * _wfr.mesh.ny)  # "flat" 2D array to take intensity data
        srwl.CalcIntFromElecField(ar1, _wfr, 6, 0, 3, _wfr.mesh.eStart, 0, 0)
        arxx = numpy.array(ar1)
        arxx = arxx.reshape((_wfr.mesh.ny, _wfr.mesh.nx))#.T
        arxx = numpy.sqrt(arxx)
        _dump_arr_2_hdf5(arxx,"amplitude/wfr_amplitude", _filename, _subgroupname)

    if _phase:
        print(">>>>>>>> writing phase...")
        # s
        ar1 = array('d', [0] * _wfr.mesh.nx * _wfr.mesh.ny)  # "flat" 2D array to take intensity data
        srwl.CalcIntFromElecField(ar1, _wfr, 0, 4, 3, _wfr.mesh.eStart, 0, 0)
        arxx = numpy.array(ar1)
        arxx = arxx.reshape((_wfr.mesh.ny, _wfr.mesh.nx))#.T


        # p
        ar2 = array('d', [0] * _wfr.mesh.nx * _wfr.mesh.ny)  # "flat" 2D array to take intensity data
        srwl.CalcIntFromElecField(ar2, _wfr, 1, 4, 3, _wfr.mesh.eStart, 0, 0)
        aryy = numpy.array(ar2)
        aryy = arxx.reshape((_wfr.mesh.ny, _wfr.mesh.nx))#.T

        _dump_arr_2_hdf5(arxx-aryy, "phase/wfr_phase", _filename, _subgroupname) # difference
        _dump_arr_2_hdf5(arxx, "phase/wfr_phase_s", _filename, _subgroupname)
        _dump_arr_2_hdf5(aryy, "phase/wfr_phase_p", _filename, _subgroupname)

    if (_complex_amplitude) or (_intensity):
        print(dir(_wfr))
        print(">>>>>>>> writing complex amplitude or intensity...",_wfr.mesh.nx, _wfr.mesh.ny, _wfr.mesh.ne)
        x_polarization = _SRW_2_Numpy(_wfr.arEx, _wfr.mesh.nx, _wfr.mesh.ny, _wfr.mesh.ne)   # sigma
        print(">>>>>>>> writing complex amplitude or intensity 1...")
        y_polarization = _SRW_2_Numpy(_wfr.arEy, _wfr.mesh.nx, _wfr.mesh.ny, _wfr.mesh.ne)   # pi

        e_field = numpy.concatenate((x_polarization, y_polarization), 3)

        print(">>>>>>>> writing complex amplitude or intensity 2...")
        complex_amplitude_s = e_field[0,:,:,0]
        complex_amplitude_p = e_field[0,:,:,1]

        print(">>>>>>>> writing complex amplitude or intensity 3...")
        if _complex_amplitude:
            _dump_arr_2_hdf5(complex_amplitude_s.T, "wfr_complex_amplitude_s", _filename, _subgroupname)
            _dump_arr_2_hdf5(complex_amplitude_p.T, "wfr_complex_amplitude_p", _filename, _subgroupname)

        print(">>>>>>>> writing complex amplitude or intensity 4...")
        if _intensity:
            intens_p = numpy.abs(complex_amplitude_p) ** 2
            intens_s = numpy.abs(complex_amplitude_s) ** 2
            intens = intens_s + intens_p

            _dump_arr_2_hdf5(intens.T,"intensity/wfr_intensity",_filename, _subgroupname)
            _dump_arr_2_hdf5(intens_s.T,"intensity/wfr_intensity_s",_filename, _subgroupname)
            _dump_arr_2_hdf5(intens_p.T,"intensity/wfr_intensity_p",_filename, _subgroupname)

    f = h5py.File(_filename, 'a')
    f1 = f[_subgroupname]

    print(">>>>>>>> writing attributes...")
    # points to the default data to be plotted
    f1.attrs['NX_class'] = 'NXentry'
    f1.attrs['default']  = 'intensity'

    #f1["wfr_method"] = "SRW"
    f1["wfr_photon_energy"] = float(_wfr.mesh.eStart)
    f1["wfr_zStart"] = _wfr.mesh.zStart
    f1["wfr_Rx_dRx"] =  numpy.array([_wfr.Rx,_wfr.dRx])
    f1["wfr_Ry_dRy"] =  numpy.array([_wfr.Ry,_wfr.dRy])
    f1["wfr_mesh_X"] =  numpy.array([_wfr.mesh.xStart,_wfr.mesh.xFin,_wfr.mesh.nx])
    f1["wfr_mesh_Y"] =  numpy.array([_wfr.mesh.yStart,_wfr.mesh.yFin,_wfr.mesh.ny])

    # Add NX plot attribites for automatic plot with silx view
    myflags = [_intensity,_amplitude,_phase]
    mylabels = ['intensity','amplitude','phase']
    for i,label in enumerate(mylabels):
        if myflags[i]:

            f2 = f1[mylabels[i]]
            f2.attrs['NX_class'] = 'NXdata'
            f2.attrs['signal'] = 'wfr_%s'%(mylabels[i])
            f2.attrs['axes'] = [b'axis_y', b'axis_x']

            # ds = nxdata.create_dataset('image_data', data=data)
            f3 = f2["wfr_%s"%(mylabels[i])]
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

    FileName = _filename.split("/")
    print(">>>> save_wfr_2_hdf5: file written/updated %s" %FileName[-1])



    # try:
    #     if not os.path.isfile(_filename):  # if file doesn't exist, create it.
    #         sys.stdout.flush()
    #         f = h5py.File(_filename, 'w')
    #         # points to the default data to be plotted
    #         f.attrs['default']          = 'entry'
    #         # give the HDF5 root some more attributes
    #         f.attrs['file_name']        = _filename
    #         f.attrs['file_time']        = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    #         f.attrs['creator']          = 'save_wfr_2_hdf5'
    #         f.attrs['code']             = 'SRW'
    #         f.attrs['HDF5_Version']     = h5py.version.hdf5_version
    #         f.attrs['h5py_version']     = h5py.version.version
    #         f.close()
    #
    #     # TODO: delete
    #     if _amplitude:
    #         print(">>>>>>>> writing amplitude...")
    #         ar1 = array('f', [0] * _wfr.mesh.nx * _wfr.mesh.ny)  # "flat" 2D array to take intensity data
    #         srwl.CalcIntFromElecField(ar1, _wfr, 6, 0, 3, _wfr.mesh.eStart, 0, 0)
    #         arxx = numpy.array(ar1)
    #         arxx = arxx.reshape((_wfr.mesh.ny, _wfr.mesh.nx))#.T
    #         arxx = numpy.sqrt(arxx)
    #         _dump_arr_2_hdf5(arxx,"amplitude/wfr_amplitude", _filename, _subgroupname)
    #
    #     if _phase:
    #         print(">>>>>>>> writing phase...")
    #         # s
    #         ar1 = array('d', [0] * _wfr.mesh.nx * _wfr.mesh.ny)  # "flat" 2D array to take intensity data
    #         srwl.CalcIntFromElecField(ar1, _wfr, 0, 4, 3, _wfr.mesh.eStart, 0, 0)
    #         arxx = numpy.array(ar1)
    #         arxx = arxx.reshape((_wfr.mesh.ny, _wfr.mesh.nx))#.T
    #
    #
    #         # p
    #         ar2 = array('d', [0] * _wfr.mesh.nx * _wfr.mesh.ny)  # "flat" 2D array to take intensity data
    #         srwl.CalcIntFromElecField(ar2, _wfr, 1, 4, 3, _wfr.mesh.eStart, 0, 0)
    #         aryy = numpy.array(ar2)
    #         aryy = arxx.reshape((_wfr.mesh.ny, _wfr.mesh.nx))#.T
    #
    #         _dump_arr_2_hdf5(arxx-aryy, "phase/wfr_phase", _filename, _subgroupname) # difference
    #         _dump_arr_2_hdf5(arxx, "phase/wfr_phase_s", _filename, _subgroupname)
    #         _dump_arr_2_hdf5(aryy, "phase/wfr_phase_p", _filename, _subgroupname)
    #
    #     if (_complex_amplitude) or (_intensity):
    #         print(">>>>>>>> writing complex amplitude or intensity...",_wfr.mesh.nx, _wfr.mesh.ny, _wfr.mesh.ne)
    #         x_polarization = _SRW_2_Numpy(_wfr.arEx, _wfr.mesh.nx, _wfr.mesh.ny, _wfr.mesh.ne)   # sigma
    #         print(">>>>>>>> writing complex amplitude or intensity 1...",x_polarization)
    #         y_polarization = _SRW_2_Numpy(_wfr.arEy, _wfr.mesh.nx, _wfr.mesh.ny, _wfr.mesh.ne)   # pi
    #
    #         e_field = numpy.concatenate((x_polarization, y_polarization), 3)
    #
    #         print(">>>>>>>> writing complex amplitude or intensity 2...")
    #         complex_amplitude_s = e_field[0,:,:,0]
    #         complex_amplitude_p = e_field[0,:,:,1]
    #
    #         print(">>>>>>>> writing complex amplitude or intensity 3...")
    #         if _complex_amplitude:
    #             _dump_arr_2_hdf5(complex_amplitude_s.T, "wfr_complex_amplitude_s", _filename, _subgroupname)
    #             _dump_arr_2_hdf5(complex_amplitude_p.T, "wfr_complex_amplitude_p", _filename, _subgroupname)
    #
    #         print(">>>>>>>> writing complex amplitude or intensity 4...")
    #         if _intensity:
    #             intens_p = numpy.abs(complex_amplitude_p) ** 2
    #             intens_s = numpy.abs(complex_amplitude_s) ** 2
    #             intens = intens_s + intens_p
    #
    #             _dump_arr_2_hdf5(intens.T,"intensity/wfr_intensity",_filename, _subgroupname)
    #             _dump_arr_2_hdf5(intens_s.T,"intensity/wfr_intensity_s",_filename, _subgroupname)
    #             _dump_arr_2_hdf5(intens_p.T,"intensity/wfr_intensity_p",_filename, _subgroupname)
    #
    #     f = h5py.File(_filename, 'a')
    #     f1 = f[_subgroupname]
    #
    #     print(">>>>>>>> writing attributes...")
    #     # points to the default data to be plotted
    #     f1.attrs['NX_class'] = 'NXentry'
    #     f1.attrs['default']  = 'intensity'
    #
    #     #f1["wfr_method"] = "SRW"
    #     f1["wfr_photon_energy"] = float(_wfr.mesh.eStart)
    #     f1["wfr_zStart"] = _wfr.mesh.zStart
    #     f1["wfr_Rx_dRx"] =  numpy.array([_wfr.Rx,_wfr.dRx])
    #     f1["wfr_Ry_dRy"] =  numpy.array([_wfr.Ry,_wfr.dRy])
    #     f1["wfr_mesh_X"] =  numpy.array([_wfr.mesh.xStart,_wfr.mesh.xFin,_wfr.mesh.nx])
    #     f1["wfr_mesh_Y"] =  numpy.array([_wfr.mesh.yStart,_wfr.mesh.yFin,_wfr.mesh.ny])
    #
    #     # Add NX plot attribites for automatic plot with silx view
    #     myflags = [_intensity,_amplitude,_phase]
    #     mylabels = ['intensity','amplitude','phase']
    #     for i,label in enumerate(mylabels):
    #         if myflags[i]:
    #
    #             f2 = f1[mylabels[i]]
    #             f2.attrs['NX_class'] = 'NXdata'
    #             f2.attrs['signal'] = 'wfr_%s'%(mylabels[i])
    #             f2.attrs['axes'] = [b'axis_y', b'axis_x']
    #
    #             # ds = nxdata.create_dataset('image_data', data=data)
    #             f3 = f2["wfr_%s"%(mylabels[i])]
    #             f3.attrs['interpretation'] = 'image'
    #
    #             # X axis data
    #             ds = f2.create_dataset('axis_y', data=1e6*numpy.linspace(_wfr.mesh.yStart,_wfr.mesh.yFin,_wfr.mesh.ny))
    #             # f1['axis1_name'] = numpy.arange(_wfr.mesh.ny)
    #             ds.attrs['units'] = 'microns'
    #             ds.attrs['long_name'] = 'Y Pixel Size (microns)'    # suggested X axis plot label
    #             #
    #             # Y axis data
    #             ds = f2.create_dataset('axis_x', data=1e6*numpy.linspace(_wfr.mesh.xStart,_wfr.mesh.xFin,_wfr.mesh.nx))
    #             ds.attrs['units'] = 'microns'
    #             ds.attrs['long_name'] = 'X Pixel Size (microns)'    # suggested Y axis plot label
    #     f.close()
    #
    #     FileName = _filename.split("/")
    #     print(">>>> save_wfr_2_hdf5: file written/updated %s" %FileName[-1])
    #
    # except:
    #     if _overwrite is not True:
    #         print(">>>> Bad input argument")
    #         return
    #     os.remove(_filename)
    #     FileName = _filename.split("/")
    #
    #     print(">>>> save_wfr_2_hdf5: file deleted %s"%FileName[-1])
    #     save_wfr_2_hdf5(_wfr,_filename,_subgroupname,_complex_amplitude,_intensity,_amplitude,_phase,_overwrite = False)

def save_stokes_2_hdf5(_Stokes,_filename,_subgroupname="wfr",_S0=True,_S1=False,_S2=False,_S3=False,_overwrite=True):
    """
     Auxiliary function to write the Stokes parameters data into a hdf5 generic file. The Stokes parameters of a plane
     monochromatic wave are four quantities: S0, S1, S2 and S3. Only three of them are independent, since they are
     related by the identity: S0^2 = S1^2 + S2^2 + S3^2.
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
            # points to the default data to be plotted
            f.attrs['default']          = 'entry'
            # give the HDF5 root some more attributes
            f.attrs['file_name']        = _filename
            f.attrs['file_time']        = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
            f.attrs['creator']          = 'save_stokes_2_hdf5'
            f.attrs['code']             = 'SRW'
            f.attrs['HDF5_Version']     = h5py.version.hdf5_version
            f.attrs['h5py_version']     = h5py.version.version
            f.close()

        arxx = numpy.array(_Stokes.arS)
        arxx = arxx.reshape((4, _Stokes.mesh.ny, _Stokes.mesh.nx)).T

        if _S0:
            _dump_arr_2_hdf5(arxx[:, :, 0].T, "Stokes_S0/S0", _filename, _subgroupname)
        if _S1:
            _dump_arr_2_hdf5(arxx[:, :, 1].T, "Stokes_S1/S1", _filename, _subgroupname)
        if _S2:
            _dump_arr_2_hdf5(arxx[:, :, 2].T, "Stokes_S2/S2", _filename, _subgroupname)
        if _S3:
            _dump_arr_2_hdf5(arxx[:, :, 3].T, "Stokes_S3/S3", _filename, _subgroupname)

        f = h5py.File(_filename, 'a')
        f1 = f[_subgroupname]

        # points to the default data to be plotted
        f1.attrs['NX_class'] = 'NXentry'
        f1.attrs['default']  = 'S0'

        #f1["Stokes_method"] = "SRW"
        f1["Stokes_photon_energy"] = _Stokes.mesh.eStart
        f1["Stokes_mesh_X"] = numpy.array([_Stokes.mesh.xStart, _Stokes.mesh.xFin, _Stokes.mesh.nx])
        f1["Stokes_mesh_Y"] = numpy.array([_Stokes.mesh.yStart, _Stokes.mesh.yFin, _Stokes.mesh.ny])

        # Add NX plot attribites for automatic plot with silx view
        myflags = [_S0,_S1,_S2,_S3]
        mylabels = ['S0','S1','S2','S3']
        for i,label in enumerate(mylabels):
            if myflags[i]:

                f2 = f1['Stokes_%s'%(mylabels[i])]
                f2.attrs['NX_class'] = 'NXdata'
                f2.attrs['signal'] = '%s'%(mylabels[i])
                f2.attrs['axes'] = [b'axis_y', b'axis_x']

                # ds = nxdata.create_dataset('image_data', data=data)
                f3 = f2["%s"%(mylabels[i])]
                f3.attrs['interpretation'] = 'image'

                # X axis data
                ds = f2.create_dataset('axis_y', data=1e6*numpy.linspace(_Stokes.mesh.yStart,_Stokes.mesh.yFin,_Stokes.mesh.ny))
                # f1['axis1_name'] = numpy.arange(_wfr.mesh.ny)
                ds.attrs['units'] = 'microns'
                ds.attrs['long_name'] = 'Y Pixel Size (microns)'    # suggested X axis plot label
                #
                # Y axis data
                ds = f2.create_dataset('axis_x', data=1e6*numpy.linspace(_Stokes.mesh.xStart,_Stokes.mesh.xFin,_Stokes.mesh.nx))
                ds.attrs['units'] = 'microns'
                ds.attrs['long_name'] = 'X Pixel Size (microns)'    # suggested Y axis plot label
        f.close()

        FileName = _filename.split("/")
        print(">>>> save_stokes_2_hdf5: file witten/updated %s" %FileName[-1])

    except:
        if _overwrite is not True:
            print(">>>> Bad input argument")
            return
        os.remove(_filename)
        FileName = _filename.split("/")
        print(">>>> save_stokes_2_hdf5: file deleted %s"%FileName[-1])
        save_stokes_2_hdf5(_Stokes,_filename,_subgroupname,_S0,_S1,_S2,_S3,_overwrite = False)

def SRWdat_2_h5(_file_path,_num_type='f'):
    """
    Auxiliary to be convert output files from SRW .dat format into a generic wavefront hdf5 generic file. Read-in tabulated
    Intensity data from an ASCII file (format is defined in srwl_uti_save_intens_ascii)
    :param _file_path: path to file for saving the wavefront
    """
    file_h5 = _file_path.replace(".dat", ".h5")
    filename = file_h5.split("/")
    wfr = SRWLWfr()
    wfr.arEx, wfr.mesh = srwl_uti_read_intens_ascii(_file_path, _num_type)

    arxx = numpy.array(wfr.arEx)

    if wfr.mesh.ne > 1:
        print('>>>> No support for such .dat file')
        return
        # sys.stdout.flush()

        # f = h5py.File(file_h5, 'w')
        # Energy = numpy.linspace(wfr.mesh.eStart, wfr.mesh.eFin, num=wfr.mesh.ne)
        # arxx = arxx.reshape((wfr.mesh.ny, wfr.mesh.nx, wfr.mesh.ne, 1))
        # arxx = arxx.swapaxes(0, 2)

        # for count in range(wfr.mesh.ne):

        #     subgroupname = "wfr_%d"%count
        #     f1 = f.create_group(subgroupname)
        #     f1["converted_array"] = arxx
        #     f1["wfr_method"] = "SRW"
        #     f1["wfr_photon_energy"] = Energy[count]
        #     f1["wfr_radii"] = numpy.array([wfr.Rx, wfr.dRx, wfr.Ry, wfr.dRy])
        #     f1["wfr_mesh"] = numpy.array([wfr.mesh.xStart, wfr.mesh.xFin, wfr.mesh.nx, wfr.mesh.yStart, wfr.mesh.yFin, wfr.mesh.ny])
        # f.close()
        # return

    arxx = arxx.reshape((wfr.mesh.ny, wfr.mesh.nx))#.T

    sys.stdout.flush()

    f = h5py.File(file_h5, 'w')
    # points to the default data to be plotted
    f.attrs['default'] = 'entry'
    # give the HDF5 root some more attributes
    f.attrs['file_name'] = filename[-1]
    f.attrs['file_time'] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    f.attrs['creator'] = 'save_wfr_2_hdf5'
    f.attrs['code'] = 'SRW'
    f.attrs['HDF5_Version'] = h5py.version.hdf5_version
    f.attrs['h5py_version'] = h5py.version.version

    f1 = f.create_group("wfr")
    # f1["converted_array"] = arxx
    # f1["wfr_method"] = "SRW"

    fdata = f1.create_dataset("converted_array/array", data=arxx)

    f1["wfr_photon_energy"] = float(wfr.mesh.eStart)
    f1["wfr_zStart"] = wfr.mesh.zStart
    f1["wfr_Rx_dRx"] = numpy.array([wfr.Rx, wfr.dRx])
    f1["wfr_Ry_dRy"] = numpy.array([wfr.Ry, wfr.dRy])
    f1["wfr_mesh_X"] = numpy.array([wfr.mesh.xStart, wfr.mesh.xFin, wfr.mesh.nx])
    f1["wfr_mesh_Y"] = numpy.array([wfr.mesh.yStart, wfr.mesh.yFin, wfr.mesh.ny])



    f2 = f1["converted_array"]
    f2.attrs['NX_class'] = 'NXdata'
    f2.attrs['signal'] = 'array'
    f2.attrs['axes'] = [b'axis_y', b'axis_x']

    # ds = nxdata.create_dataset('image_data', data=data)
    f3 = f2['array']
    f3.attrs['interpretation'] = 'image'

    # X axis data
    ds = f2.create_dataset('axis_y', data=1e6 * numpy.linspace(wfr.mesh.yStart, wfr.mesh.yFin, wfr.mesh.ny))
    # f1['axis1_name'] = numpy.arange(_wfr.mesh.ny)
    ds.attrs['units'] = 'microns'
    ds.attrs['long_name'] = 'Y Pixel Size (microns)'  # suggested X axis plot label
    #
    # Y axis data
    ds = f2.create_dataset('axis_x', data=1e6 * numpy.linspace(wfr.mesh.xStart, wfr.mesh.xFin, wfr.mesh.nx))
    ds.attrs['units'] = 'microns'
    ds.attrs['long_name'] = 'X Pixel Size (microns)'  # suggested Y axis plot label

    f.close()

#
# Auxiliar functions
#

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
    e = e.reshape((_ny, _nx, _ne, 1))
    e = e.swapaxes(0, 2)

    return e.copy()

def _dump_arr_2_hdf5(_arr,_calculation, _filename, _subgroupname):
    """
    Auxiliary routine to save_wfr_2_hdf5() and save_stokes_2_hdf5()
    :param _arr: (usually 2D) array to be saved on the hdf5 file inside the _subgroupname
    :param _calculation
    :param _filename: path to file for saving the wavefront
    :param _subgroupname: container mechanism by which HDF5 files are organised
    """
    sys.stdout.flush()
    f = h5py.File(_filename, 'a')
    try:
        f1 = f.create_group(_subgroupname)
    except:
        f1 = f[_subgroupname]
    #f1[_calculation] = _arr
    fdata = f1.create_dataset(_calculation, data=_arr)
    f.close()

def load_h5_file_to_dictionary(filename,filepath):

    try:
        f = h5py.File(filename, 'r')
        # mesh_X = f[filepath+"/wfr_mesh_X"].value
        # mesh_Y = f[filepath+"/wfr_mesh_Y"].value
        # complex_amplitude_s = f[filepath+"/wfr_complex_amplitude_s"].value.T
        # complex_amplitude_p = f[filepath+"/wfr_complex_amplitude_p"].value.T
        # photon_energy = f[filepath+"/wfr_photon_energy"].value

        out =  {
            "wfr_complex_amplitude_s":f[filepath+"/wfr_complex_amplitude_s"].value.T,
            "wfr_complex_amplitude_p":f[filepath+"/wfr_complex_amplitude_p"].value.T,
            "wfr_photon_energy":f[filepath+"/wfr_photon_energy"].value,
            "wfr_zStart":f[filepath+"/wfr_zStart"].value,
            "wfr_mesh_X":f[filepath+"/wfr_mesh_X"].value,
            "wfr_mesh_Y":f[filepath+"/wfr_mesh_Y"].value,
            "wfr_Rx_dRx":f[filepath+"/wfr_Rx_dRx"].value,
            "wfr_Ry_dRy":f[filepath+"/wfr_Ry_dRy"].value,
        }

        f.close()
        return out
    except:
        raise Exception("Failed to load SRW wavefront to h5 file: "+filename)

def dictionary_to_srw_wavefront(wdic):
    """
    Creates a SRWWavefront from pi and sigma components of the electrical field.
    :param horizontal_start: Horizontal start position of the grid in m
    :param horizontal_end: Horizontal end position of the grid in m
    :param horizontal_efield: The pi component of the complex electrical field
    :param vertical_start: Vertical start position of the grid in m
    :param vertical_end: Vertical end position of the grid in m
    :param vertical_efield: The sigma component of the complex electrical field
    :param energy: Energy in eV
    :param z: z position of the wavefront in m
    :param Rx: Instantaneous horizontal wavefront radius
    :param dRx: Error in instantaneous horizontal wavefront radius
    :param Ry: Instantaneous vertical wavefront radius
    :param dRy: Error in instantaneous vertical wavefront radius
    :return: A wavefront usable with SRW.
    """

    w_s = wdic["wfr_complex_amplitude_s"]
    w_p = wdic["wfr_complex_amplitude_p"]
    energy = wdic["wfr_photon_energy"]
    X = wdic["wfr_mesh_X"]
    Y = wdic["wfr_mesh_Y"]
    RX = wdic["wfr_Rx_dRx"]
    RY = wdic["wfr_Ry_dRy"]
    Z = wdic["wfr_zStart"]


    horizontal_size = w_s.shape[0]
    vertical_size = w_s.shape[1]

    if horizontal_size % 2 == 1 or \
       vertical_size % 2 == 1:
        # raise Exception("Both horizontal and vertical grid must have even number of points")
        print("NumpyToSRW: WARNING: Both horizontal and vertical grid must have even number of points")

    horizontal_field = numpyArrayToSRWArray(w_s)
    vertical_field = numpyArrayToSRWArray(w_p)

    srw_wavefront = SRWLWfr(_arEx=horizontal_field,
                            _arEy=vertical_field,
                            _typeE='f',
                            _eStart=energy,
                            _eFin=energy,
                            _ne=1,
                            _xStart=X[0],
                            _xFin=X[1],
                            _nx=int(X[2]),
                            _yStart=Y[0],
                            _yFin=Y[1],
                            _ny=int(Y[2]),
                            _zStart=Z)

    srw_wavefront.Rx = RX[0]
    srw_wavefront.Ry = RY[0]
    srw_wavefront.dRx = RX[1]
    srw_wavefront.dRy = RY[1]

    return srw_wavefront

def load_h5_file_to_srw_wavefront(filename,filepath):
    wdic = load_h5_file_to_dictionary(filename,filepath)
    wfr = dictionary_to_srw_wavefront(wdic)
    return wfr