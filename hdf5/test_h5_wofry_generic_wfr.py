import time
import sys
import numpy
import os
from numpy.testing import assert_array_almost_equal

import h5py

from srwlibAux import _dump_arr_2_hdf5



def save_wofry_wavefront_to_hdf5(wfr,filename,subgroupname="wfr",intensity=False,phase=False,overwrite=True):
    """
    Auxiliary function to write wavefront data into a hdf5 generic file.
    When using the append mode to write h5 files, overwriting does not work and makes the code crash. To avoid this
    issue, try/except is used. If by any chance a file should be overwritten, it is firstly deleted and re-written.
    :param wfr: input / output resulting Wavefront structure (instance of SRWLWfr);
    :param filename: path to file for saving the wavefront
    :param subgroupname: container mechanism by which HDF5 files are organised
    :param intensity: writes intensity for sigma and pi polarisation (default=False)
    :param amplitude: Single-Electron" Intensity - total polarisation (instance of srwl.CalcIntFromElecField)
    :param phase: "Single-Electron" Radiation Phase - total polarisation (instance of srwl.CalcIntFromElecField)
    :param overwrite: flag that should always be set to True to avoid infinity loop on the recursive part of the function.
    """

    try:
        if not os.path.isfile(filename):  # if file doesn't exist, create it.
            sys.stdout.flush()
            f = h5py.File(filename, 'w')
            # point to the default data to be plotted
            f.attrs['default']          = 'entry'
            # give the HDF5 root some more attributes
            f.attrs['file_name']        = filename
            f.attrs['file_time']        = time.time()
            f.attrs['creator']          = 'save_wofry_wavefront_to_hdf5'
            f.attrs['HDF5_Version']     = h5py.version.hdf5_version
            f.attrs['h5py_version']     = h5py.version.version
            f.close()

        # always writes complex amplitude
        # if _complex_amplitude:
        x_polarization = wfr.get_complex_amplitude()       # sigma
        y_polarization = wfr.get_complex_amplitude()*0.0   # pi

        _dump_arr_2_hdf5(x_polarization, "wfr_complex_amplitude_sigma", filename, subgroupname)
        _dump_arr_2_hdf5(y_polarization, "wfr_complex_amplitude_pi", filename, subgroupname)


        if intensity:
            _dump_arr_2_hdf5(wfr.get_intensity().T,"intensity/wfr_intensity_transposed", filename, subgroupname)

        if phase:
            _dump_arr_2_hdf5(wfr.get_phase().T,"phase/wfr_phase_transposed", filename, subgroupname)

        # add mesh and SRW information
        f = h5py.File(filename, 'a')
        f1 = f[subgroupname]

        # point to the default data to be plotted
        f1.attrs['NX_class'] = 'NXentry'
        f1.attrs['default']          = 'intensity'

        f1["wfr_method"] = "WOFRY"
        f1["wfr_photon_energy"] = wfr.get_photon_energy()
        x = wfr.get_coordinate_x()
        y = wfr.get_coordinate_y()

        f1["wfr_mesh"] =  numpy.array([x[0],x[-1],x.size,y[0],y[-1],y.size])

        # Add NX plot attribites for automatic plot with silx view
        myflags = [intensity,phase]
        mylabels = ['intensity','phase']
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
                ds = f2.create_dataset('axis_y', data=1e6*y)
                # f1['axis1_name'] = numpy.arange(_wfr.mesh.ny)
                ds.attrs['units'] = 'microns'
                ds.attrs['long_name'] = 'Y Pixel Size (microns)'    # suggested X axis plot label
                #
                # Y axis data
                ds = f2.create_dataset('axis_x', data=1e6*x)
                ds.attrs['units'] = 'microns'
                ds.attrs['long_name'] = 'X Pixel Size (microns)'    # suggested Y axis plot label
        f.close()

    except:
        # TODO: check exit??
        if overwrite is not True:
            print(">>>> Bad input argument")
            sys.exit()
        os.remove(filename)
        print(">>>> save_wfr_2_hdf5: file deleted %s"%filename)

        FileName = filename.split("/")
        # print(">>>> save_wfr_2_hdf5: %s"%_subgroupname+" in %s was deleted." %FileName[-1])
        save_wofry_wavefront_to_hdf5(wfr,filename,subgroupname, intensity=intensity, phase=phase, overwrite=False)

    print(">>>> save_wfr_2_hdf5: witten/updated %s data in file: %s"%(subgroupname,filename))

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

    print(">>>> sigma shapes: ", wfr.size())

    assert_array_almost_equal(wfr.get_complex_amplitude(),amplitude_sigma)



if __name__ == "__main__":

    from wofry.propagator.wavefront2D.generic_wavefront import GenericWavefront2D

    wfr = GenericWavefront2D.initialize_wavefront_from_range(-0.002,0.002,-0.001,0.001,(200,200))
    wfr.set_gaussian(0.002/6,0.001/12)
    save_wofry_wavefront_to_hdf5(wfr,"tmp_wofry.h5",subgroupname="wfr", intensity=True,phase=True,overwrite=True)


    wfr.set_gaussian(0.002/6/2,0.001/12/2)
    save_wofry_wavefront_to_hdf5(wfr,"tmp_wofry.h5",subgroupname="wfr2", intensity=True,phase=False,overwrite=False)

    test_same_amplitudes(wfr,"tmp_wofry.h5","wfr2")

